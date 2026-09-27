"""Software invariants only. These fixtures are not training or accuracy evidence."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
try:
    import numpy as np
    import torch
    from ml import temporal as t
    HAS_TRAINING = True
except ModuleNotFoundError:
    HAS_TRAINING = False


def session():
    landmarks=[dict(x=.5,y=.5,visibility=.99) for _ in range(33)]
    for i,x,y in [(11,.4,.3),(12,.6,.3),(23,.4,.7),(24,.6,.7),(13,.35,.42),(15,.4,.28),(14,.65,.42),(16,.6,.28)]:
        landmarks[i]=dict(x=x,y=y,visibility=.99)
    frames=[dict(t=i*40,width=1280,height=720,landmarks=copy.deepcopy(landmarks),inferenceMs=1) for i in range(151)]
    return dict(schemaVersion='1.0',id='fixture',source='file',model='full',stance='orthodox',durationMs=6000,frames=frames,events=[],annotationsComplete=True,
                annotations=[dict(id=str(i),label='jab' if i%2==0 else 'cross',hand='left' if i%2==0 else 'right',startMs=500+i*1000,endMs=1000+i*1000) for i in range(4)])


@unittest.skipUnless(HAS_TRAINING, 'optional training dependencies not installed')
class TemporalTests(unittest.TestCase):
    def test_network_never_uses_future_frames(self):
        torch.manual_seed(1);model=t.CausalTCN().eval();x=torch.randn(1,len(t.FEATURE_NAMES),50);changed=x.clone();changed[:,:,25:]+=100
        with torch.no_grad():
            np.testing.assert_allclose(model(x)[:,:25].numpy(),model(changed)[:,:25].numpy(),atol=1e-6)
            np.testing.assert_allclose(model(x)[:,:25].numpy(),model(x[:,:,:25]).numpy(),atol=1e-6)

    def test_features_are_causal_and_mask_missing_coordinates_without_fill(self):
        a=session();b=copy.deepcopy(a);b['frames'][20]['landmarks'][15]['visibility']=.1;b['frames'][20]['landmarks'][15]['x']=.99
        x,_,_=t.feature_series(a,'left');y,_,_=t.feature_series(b,'left');np.testing.assert_array_equal(x[:20],y[:20])
        for feature in ['wrist_x','wrist_y','wrist_observed','wrist_vx','wrist_vy','arm_valid']:
            self.assertEqual(y[20,t.FEATURE_NAMES.index(feature)],0)
        self.assertEqual(y[21,t.FEATURE_NAMES.index('wrist_vx')],0)

    def test_gap_resets_features_and_temporal_context(self):
        a=session();a['frames']=a['frames'][:10]+a['frames'][40:50]
        x,_,segments=t.feature_series(a,'left');self.assertEqual(segments,[(0,10),(10,20)])
        self.assertEqual(x[10,t.FEATURE_NAMES.index('dt')],0)
        model=t.CausalTCN().eval();mean=np.zeros(len(t.FEATURE_NAMES),np.float32);std=np.ones_like(mean)
        full=t.predict_probabilities(model,mean,std,a)['left'];tail=copy.deepcopy(a);tail['frames']=tail['frames'][10:]
        np.testing.assert_allclose(full[10:],t.predict_probabilities(model,mean,std,tail)['left'],atol=1e-6)

    def test_unknown_intervals_are_not_background_targets(self):
        s=session();s['annotations'].append(dict(id='unknown',label='unobservable',hand='unknown',startMs=1000,endMs=2000))
        y,mask=t.targets(s,np.array([0,750,1250,2500]),'left');self.assertEqual(y.tolist(),[0,1,0,1]);self.assertEqual(mask.tolist(),[1,1,0,1])
        s['annotationsComplete']=False
        with self.assertRaisesRegex(ValueError,'complete'):t.targets(s,np.array([0]),'left')

    def test_entire_session_roles_are_disjoint(self):
        for n in [3,5]:
            splits=t.session_splits(n);self.assertEqual(sorted(s['test'] for s in splits),list(range(n)))
            for s in splits:self.assertEqual(len(set(s['train']+[s['validation'],s['test']])),n)
        with self.assertRaises(ValueError):t.session_splits(2)

    def test_duplicate_capture_rejected_even_if_id_and_telemetry_change(self):
        s=session();other=copy.deepcopy(s);other['id']='different';other['frames'][0]['inferenceMs']=10
        self.assertEqual(t.capture_fingerprint(s),t.capture_fingerprint(other))
        with tempfile.TemporaryDirectory() as d:
            paths=[]
            for i,value in enumerate([s,other]):
                p=Path(d)/f'{i}.json';p.write_text(json.dumps(value));paths.append(p)
            with self.assertRaisesRegex(ValueError,'Duplicate'):t.read_training_sessions(paths)

    def test_decoder_is_causal_preserves_hand_and_does_not_complete_tail(self):
        times=np.arange(0,600,40);high=np.array([0,0,.8,.9,.9,.9,.9,0,0,0,0,0,0,0,0]);zero=np.zeros_like(high)
        events=t.decode_events(times,dict(left=high,right=zero),'southpaw');self.assertEqual(len(events),1)
        self.assertEqual((events[0]['hand'],events[0]['label']),('left','cross'));self.assertEqual(events[0]['detectedAtMs'],320)
        self.assertEqual(t.decode_events(times[:7],dict(left=high[:7],right=zero[:7]),'orthodox'),[])

    def test_decoder_does_not_join_source_gaps_or_count_single_spikes(self):
        times=[0,40,80,120,500,540];p=[0,.9,.9,.9,0,0]
        self.assertEqual(t.decode_events(times,dict(left=p,right=p),'orthodox'),[])
        self.assertEqual(t.decode_events([0,40,80,120],dict(left=[0,.99,0,0],right=[0]*4),'orthodox'),[])

    def test_checkpoint_roundtrip_and_no_overwrite(self):
        model=t.CausalTCN().eval();mean=np.zeros(len(t.FEATURE_NAMES),np.float32);std=np.ones_like(mean)
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'weights.json';t.save_checkpoint(path,model,mean,std,[],'a'*64);loaded,mu,sd,_=t.load_checkpoint(path)
            a=t.predict_probabilities(model,mean,std,session());b=t.predict_probabilities(loaded,mu,sd,session())
            np.testing.assert_array_equal(a['left'],b['left'])
            with self.assertRaises(FileExistsError):t.save_checkpoint(path,model,mean,std,[],'a'*64)

    def test_annotation_changes_cannot_change_predictions(self):
        s=session();model=t.CausalTCN().eval();mean=np.zeros(len(t.FEATURE_NAMES),np.float32);std=np.ones_like(mean)
        before=t.predict_probabilities(model,mean,std,s);s['annotations']=[];s['annotationsComplete']=False
        after=t.predict_probabilities(model,mean,std,s);np.testing.assert_array_equal(before['right'],after['right'])

if __name__=='__main__':unittest.main()
