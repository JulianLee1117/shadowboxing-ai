import { useId, useState, type ReactNode } from "react";

/** Small explicit choices, with no hidden menu or custom keyboard contract. */
export function ChoiceGroup<T extends string | number>({
  label,
  value,
  options,
  onChange,
  disabled = false,
}: {
  label: string;
  value: T;
  options: readonly { value: T; label: string }[];
  onChange: (value: T) => void;
  disabled?: boolean;
}) {
  return (
    <div className="choice-field">
      <span className="choice-label">{label}</span>
      <div className="choice-group" role="group" aria-label={label}>
        {options.map((option) => (
          <button
            type="button"
            key={option.value}
            aria-pressed={value === option.value}
            disabled={disabled}
            onClick={() => onChange(option.value)}
          >
            {option.label}
          </button>
        ))}
      </div>
    </div>
  );
}

/** An ordinary button opens secondary controls without a dropdown affordance. */
export function UtilityPanel({
  label,
  children,
  className = "",
}: {
  label: string;
  children: ReactNode;
  className?: string;
}) {
  const [open, setOpen] = useState(false);
  const id = useId();
  return (
    <section className={`utility-panel ${className}`}>
      <button
        className="text-button utility-toggle"
        aria-expanded={open}
        aria-controls={id}
        onClick={() => setOpen((old) => !old)}
      >
        {label}
        {open && <span aria-hidden="true"> · close</span>}
      </button>
      <div id={id} hidden={!open} className="utility-content">
        {children}
      </div>
    </section>
  );
}
