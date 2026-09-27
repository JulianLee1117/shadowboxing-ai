"""Experimental causal per-arm straight-action training on complete session exports.

CPU reference runtime; no pose training, interpolation, future context or browser claim.
"""
from __future__ import annotations
import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import random
import time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from ml.evaluate import build_report, validate_session

PROTOCOL_VERSION = 'causal-arm-tcn-1'
CONFIG = dict(seed=20260926, epochs=120, learningRate=.002, weightDecay=.001,
              channels=24, kernel=5, dilations=[1, 2], maxGapMs=200,
              onThreshold=.6, offThreshold=.4, onsetFrames=2, offsetFrames=2,
              minEventMs=120, maxEventMs=1800, confidenceFloor=.65)
FEATURE_NAMES = [f'{joint}_{v}' for joint in ('shoulder', 'elbow', 'wrist')
                 for v in ('x', 'y', 'visibility', 'presence', 'observed', 'vx', 'vy')]
FEATURE_NAMES += ['angle', 'reach', 'angle_delta', 'reach_delta', 'dt',
                  'torso_valid', 'arm_valid', 'physical_left', 'lead']


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def write_new(path: Path, value) -> None:
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def valid(point: dict) -> bool:
    return (all(isinstance(point.get(k), (int, float)) and math.isfinite(point[k]) for k in ('x', 'y', 'visibility'))
            and 0 <= point['x'] <= 1 and 0 <= point['y'] <= 1
            and point['visibility'] >= CONFIG['confidenceFloor']
            and math.isfinite(point.get('presence', 1)) and point.get('presence', 1) >= .5)


def feature_series(session: dict, hand: str) -> tuple[np.ndarray, np.ndarray, list[tuple[int, int]]]:
    """One feature row per actual observation; invalid coordinates are zero+masked.

    First derivatives use only two observed valid samples with source gap<=200ms.
    Recurrent context must reset at returned segment boundaries. No resampling.
    """
    rows, times, segments = [], [], []
    previous = None
    start = 0
    for index, frame in enumerate(session['frames']):
        t = frame['t']; dt = 0 if previous is None else t - previous['t']
        if previous is not None and (dt <= 0 or dt > CONFIG['maxGapMs'] or (frame['width'], frame['height']) != previous['dimensions']):
            segments.append((start, index)); start = index; previous = None; dt = 0
        points = frame['landmarks']; aspect = frame['width'] / frame['height']
        at = lambda i: points[i] if i < len(points) else {}
        anchors = [at(i) for i in (11, 12, 23, 24)]
        torso_ok = all(valid(p) for p in anchors)
        scale = 0
        if torso_ok:
            upper = np.array([(anchors[0]['x'] + anchors[1]['x']) * aspect / 2,
                              (anchors[0]['y'] + anchors[1]['y']) / 2])
            lower = np.array([(anchors[2]['x'] + anchors[3]['x']) * aspect / 2,
                              (anchors[2]['y'] + anchors[3]['y']) / 2])
            scale = float(np.linalg.norm(upper-lower)); torso_ok = scale >= .08
        else:
            upper = np.zeros(2)
        raw = [at(i) for i in ((11, 13, 15) if hand == 'left' else (12, 14, 16))]
        coords, observed, row = [], [], []
        for j, point in enumerate(raw):
            ok = torso_ok and valid(point)
            xy = (np.array([point['x'] * aspect, point['y']]) - upper) / scale if ok else np.zeros(2)
            xy = np.clip(xy, -4, 4)
            velocity = (xy - previous['coords'][j]) / (dt / 1000) if (previous is not None and dt > 0 and ok and previous['observed'][j]) else np.zeros(2)
            confidence = lambda key, default: float(np.clip(point.get(key, default), 0, 1)) if isinstance(point.get(key, default), (int, float)) and math.isfinite(point.get(key, default)) else 0.
            row += [*xy, confidence('visibility', 0), confidence('presence', 1), float(ok), *np.clip(velocity / 10, -4, 4)]
            coords.append(xy); observed.append(ok)
        arm_ok = all(observed)
        angle = reach = 0.
        if arm_ok:
            a, b = coords[0]-coords[1], coords[2]-coords[1]
            denom = np.linalg.norm(a) * np.linalg.norm(b)
            arm_ok = denom > 1e-6
            if arm_ok:
                angle = math.acos(float(np.clip(np.dot(a,b)/denom,-1,1))) / math.pi
                reach = float(np.linalg.norm(coords[2]-coords[0]))
        delta_ok = previous is not None and dt > 0 and arm_ok and previous['arm_ok']
        row += [angle, reach, (angle-previous['angle']) if delta_ok else 0,
                (reach-previous['reach']) if delta_ok else 0, min(dt, CONFIG['maxGapMs']) / 100,
                float(torso_ok), float(arm_ok), float(hand == 'left'),
                float(hand == ('left' if session.get('stance') == 'orthodox' else 'right'))]
        rows.append(row); times.append(t)
        previous = dict(t=t, dimensions=(frame['width'], frame['height']), coords=coords, observed=observed, angle=angle, reach=reach, arm_ok=arm_ok)
    if rows: segments.append((start, len(rows)))
    return np.asarray(rows, dtype=np.float32).reshape(-1, len(FEATURE_NAMES)), np.asarray(times), segments


def targets(session: dict, times: np.ndarray, hand: str) -> tuple[np.ndarray, np.ndarray]:
    if session.get('annotationsComplete') is not True:
        raise ValueError('Temporal training requires explicit complete continuous annotations')
    y, mask = np.zeros(len(times), np.float32), np.ones(len(times), np.float32)
    for a in session['annotations']:
        inside = (times >= a['startMs']) & (times <= a['endMs'])
        if a['label'] == 'unobservable' or (a['label'] in ('jab', 'cross') and a['hand'] == 'unknown'):
            mask[inside] = 0
        elif a['hand'] == hand and a['label'] in ('jab', 'cross'):
            y[inside] = 1
    return y, mask


class CausalTCN(nn.Module):
    def __init__(self, inputs=len(FEATURE_NAMES)):
        super().__init__()
        self.conv1 = nn.Conv1d(inputs, CONFIG['channels'], CONFIG['kernel'], dilation=1)
        self.conv2 = nn.Conv1d(CONFIG['channels'], CONFIG['channels'], CONFIG['kernel'], dilation=2)
        self.head = nn.Conv1d(CONFIG['channels'], 1, 1)

    def forward(self, x):
        x = F.relu(self.conv1(F.pad(x, (4, 0))))
        x = F.relu(self.conv2(F.pad(x, (8, 0))))
        return self.head(x).squeeze(1)


def examples(session):
    result = []
    for hand in ('left', 'right'):
        x, times, segments = feature_series(session, hand); y, mask = targets(session, times, hand)
        for start, end in segments:
            result.append((x[start:end], y[start:end], mask[start:end]))
    return result


def train_model(sessions: list[dict], seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    data = [e for s in sessions for e in examples(s)]
    observed = np.concatenate([x[m > 0] for x, _, m in data])
    mean = observed.mean(0); std = np.maximum(observed.std(0), .1)
    positives = sum(float((y*m).sum()) for _, y, m in data)
    negatives = sum(float(((1-y)*m).sum()) for _, y, m in data)
    if min(positives, negatives) < 30:
        raise ValueError('Training requires >=30 known positive and negative arm-frames')
    weight = torch.tensor(min(10., negatives/positives))
    model = CausalTCN(); optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG['learningRate'], weight_decay=CONFIG['weightDecay'])
    batches = [(torch.tensor(((x-mean)/std).T[None]), torch.tensor(y[None]), torch.tensor(m[None])) for x,y,m in data]
    losses = []
    for epoch in range(CONFIG['epochs']):
        model.train(); optimizer.zero_grad(); numerator = 0.; denominator = 0.
        for x,y,m in batches:
            numerator = numerator + (F.binary_cross_entropy_with_logits(model(x), y, pos_weight=weight, reduction='none') * m).sum()
            denominator += m.sum()
        loss = numerator / denominator; loss.backward(); optimizer.step(); losses.append(float(loss.detach()))
    model.eval()
    return model, mean, std, {'trainingLossFirst':losses[0], 'trainingLossFinal':losses[-1], 'positiveWeight':float(weight), 'knownPositiveFrames':positives, 'knownNegativeFrames':negatives}


def predict_probabilities(model, mean, std, session):
    outputs = {}
    with torch.no_grad():
        for hand in ('left', 'right'):
            x,times,segments = feature_series(session,hand); values=np.zeros(len(x), np.float32)
            for start,end in segments:
                tensor=torch.tensor(((x[start:end]-mean)/std).T[None])
                values[start:end]=torch.sigmoid(model(tensor))[0].numpy()
            outputs[hand] = values
    return outputs


def decode_events(times, probabilities, stance):
    """Causal hysteresis with observed timestamps and confirmed probability offset.

    Timestamp gaps abort the pending event. An unfinished tail is never fabricated
    into a completed event. No ground-truth interval is used by inference/decoding.
    """
    result = []
    for hand in ('left','right'):
        pending=[]; active=None; off=[]; previous=None
        for t,p in zip(times, probabilities[hand]):
            t=float(t); p=float(p)
            if previous is not None and (t<=previous or t-previous>CONFIG['maxGapMs']):
                pending=[];active=None;off=[]
            previous=t
            if active is None:
                pending = pending+[(t,p)] if p>=CONFIG['onThreshold'] else []
                if len(pending)>=CONFIG['onsetFrames']:
                    active={'start':pending[0][0], 'peak':max(pending,key=lambda q:q[1]),'maximum':max(q[1] for q in pending)};pending=[]
                continue
            if t-active['start']>CONFIG['maxEventMs']:
                active=None;pending=[];off=[];continue
            if p>active['maximum']: active['maximum']=p;active['peak']=(t,p)
            off=off+[t] if p<CONFIG['offThreshold'] else []
            if len(off)>=CONFIG['offsetFrames']:
                end=off[0]
                if end-active['start']>=CONFIG['minEventMs']:
                    role='lead' if hand==('left' if stance=='orthodox' else 'right') else 'rear'
                    result.append(dict(id=f'temporal-{hand}-{len(result)+1}',hand=hand,role=role,label='jab' if role=='lead' else 'cross',startMs=active['start'],peakMs=min(end,active['peak'][0]),endMs=end,detectedAtMs=t,score=active['maximum'],extension=0,guardReturn='unassessable',experimental=True))
                active=None;off=[]
    return sorted(result,key=lambda e:e['startMs'])


def save_checkpoint(path, model, mean, std, training_fingerprints, protocol_hash):
    write_new(path,dict(format='causal-arm-tcn-json-1',protocolVersion=PROTOCOL_VERSION,config=CONFIG,features=FEATURE_NAMES,
                         mean=mean.tolist(),std=std.tolist(),weights={k:v.detach().tolist() for k,v in model.state_dict().items()},
                         trainedSessionFingerprints=training_fingerprints,protocolSha256=protocol_hash,
                         caveat='Experimental same-person development model; numeric export has no verified browser runtime.'))


def load_checkpoint(path):
    c=json.loads(Path(path).read_text())
    if c.get('format')!='causal-arm-tcn-json-1' or c.get('features')!=FEATURE_NAMES or c.get('config')!=CONFIG:
        raise ValueError('Checkpoint feature/config version mismatch')
    model=CausalTCN();model.load_state_dict({k:torch.tensor(v) for k,v in c['weights'].items()});model.eval()
    return model,np.asarray(c['mean'],np.float32),np.asarray(c['std'],np.float32),c


def capture_fingerprint(session):
    return sha(canonical([{k:f[k] for k in ('t','width','height','landmarks')} for f in session['frames']]))


def session_splits(count):
    if count < 3: raise ValueError('At least3 sessions required')
    return [dict(test=i,validation=(i+1)%count,train=[j for j in range(count) if j not in (i,(i+1)%count)]) for i in range(count)]


def read_training_sessions(paths):
    sessions=[];fingerprints=[];seen=set();ids=set()
    for path in paths:
        raw=Path(path).read_bytes();session=json.loads(raw);validate_session(session)
        if session.get('annotationsComplete') is not True: raise ValueError('Annotations must be complete')
        if session.get('stance') not in ('orthodox','southpaw'): raise ValueError('Explicit anatomical stance required')
        frame_hash=capture_fingerprint(session)
        if frame_hash in seen or session['id'] in ids: raise ValueError('Duplicate capture/session would leak across splits')
        seen.add(frame_hash);ids.add(session['id'])
        labels=[a for a in session['annotations'] if a['label'] in ('jab','cross') and a['hand'] in ('left','right')]
        if len(session['frames'])<100 or session['durationMs']<5000 or len(labels)<4 or not all(any(a['hand']==h for a in labels) for h in ('left','right')):
            raise ValueError('Each session needs >=100 frames,5s,4 labeled straights and both hands')
        sessions.append(session);fingerprints.append(dict(sessionId=session['id'],fileSha256=sha(raw),framesSha256=frame_hash,annotationsSha256=sha(canonical(session['annotations']))))
    if len(sessions)<3: raise ValueError('At least3 independent annotated sessions required for train/validation/test roles')
    return sessions,fingerprints


def run_experiment(paths, output):
    sessions,fingerprints=read_training_sessions(paths)
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    n=len(sessions)
    splits=session_splits(n)
    protocol=dict(version=PROTOCOL_VERSION,createdAt=datetime.now(timezone.utc).isoformat(),config=CONFIG,features=FEATURE_NAMES,
                  sourceSha256=sha(Path(__file__).read_bytes()),torchVersion=torch.__version__,numpyVersion=np.__version__,device='cpu',
                  fingerprints=fingerprints,splits=splits,selection='Fixed epochs and decoder; validation loss reported but never used to select checkpoint or hyperparameters.',
                  scope='Same-person/same-day exploratory session-held-out development; not unseen-person validation. C informed earlier detector work.',
                  baselineReference='Compare current deterministic detector on identical captures and annotations; that development comparator is not an independently held-out learned model.')
    write_new(output/'protocol.json',protocol);protocol_hash=sha(canonical(protocol));predictions=[];folds=[]
    for fold,split in enumerate(splits):
        before=time.perf_counter();model,mean,std,training=train_model([sessions[i] for i in split['train']],CONFIG['seed']+fold)
        checkpoint=output/f'fold-{fold}.weights.json';save_checkpoint(checkpoint,model,mean,std,[fingerprints[i] for i in split['train']],protocol_hash)
        validation=sessions[split['validation']];vp=predict_probabilities(model,mean,std,validation)
        vy=[];vv=[]
        for hand in ('left','right'):
            y,mask=targets(validation,np.array([f['t'] for f in validation['frames']]),hand);vy.extend(y[mask>0]);vv.extend(vp[hand][mask>0])
        val_loss=float(F.binary_cross_entropy(torch.tensor(np.array(vv)),torch.tensor(np.array(vy))))
        test=sessions[split['test']];p=predict_probabilities(model,mean,std,test);times=[f['t'] for f in test['frames']]
        result=copy.deepcopy(test);result['events']=decode_events(times,p,test['stance']);result['detectorVersion']=PROTOCOL_VERSION
        result['temporalProvenance']=dict(protocolSha256=protocol_hash,checkpointSha256=sha(checkpoint.read_bytes()),testFrameFingerprint=fingerprints[split['test']]['framesSha256'],trainedOnTest=False)
        write_new(output/f'fold-{fold}.test-session.json',result)
        write_new(output/f'fold-{fold}.probabilities.json',dict(times=times,**{h:v.tolist() for h,v in p.items()}))
        predictions.append(result);folds.append(dict(fold=fold,split=split,validationBinaryCrossEntropy=val_loss,seconds=time.perf_counter()-before,**training))
    report=build_report(predictions);report['temporalProtocol']=protocol;report['folds']=folds
    write_new(output/'cross-validation-report.json',report)
    model,mean,std,training=train_model(sessions,CONFIG['seed'])
    save_checkpoint(output/'final-all-development.weights.json',model,mean,std,fingerprints,protocol_hash)
    write_new(output/'final-training.json',dict(**training,evaluation='No independent test remains for this all-session checkpoint; cross-validation reports different checkpoints.'))
    print(json.dumps(dict(output=str(output),metrics=report['eventMetrics'],folds=folds),indent=2))
    return report


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    train=sub.add_parser('train');train.add_argument('paths',nargs='+',type=Path);train.add_argument('--output',required=True,type=Path)
    predict=sub.add_parser('predict');predict.add_argument('--checkpoint',required=True,type=Path);predict.add_argument('--input',required=True,type=Path);predict.add_argument('--output',required=True,type=Path)
    a=p.parse_args(argv)
    try:
        if a.command=='train': run_experiment(a.paths,a.output)
        else:
            session=json.loads(a.input.read_text());validate_session(session);model,mean,std,c=load_checkpoint(a.checkpoint)
            result=copy.deepcopy(session);probabilities=predict_probabilities(model,mean,std,session)
            result['events']=decode_events([f['t'] for f in session['frames']],probabilities,session['stance']);result['detectorVersion']=PROTOCOL_VERSION
            result['temporalProvenance']=dict(checkpointSha256=sha(a.checkpoint.read_bytes()),protocolSha256=c['protocolSha256'],trainedOnInput=any(f['framesSha256']==capture_fingerprint(session) for f in c['trainedSessionFingerprints']))
            write_new(a.output,result)
        return 0
    except (ValueError,OSError,KeyError,TypeError) as error:
        print(f'Temporal experiment failed: {error}');return 2

if __name__=='__main__': raise SystemExit(main())
