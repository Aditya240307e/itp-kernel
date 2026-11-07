-- Does Lean 4.15.0 admit the three families of dev_positivity.py?  (B0 is the control.)
inductive B0 : Type where
  | mk0 : ((B0 -> False) -> B0) -> B0

inductive Q : Prop -> Type where
  | mkq (A : Prop) (h : Q (Q A -> False)) : Q A

inductive R : Prop -> Type where
  | mkr (A : Prop) : R (R A -> False)
