-- The rules that the declaration of R would give, postulated as axioms (Lean refuses the declaration).
axiom R : Prop -> Type
axiom mkr : forall A : Prop, R (R A -> False)
axiom R_inv : forall x : Prop, R x -> Exists (fun B : Prop => (R B -> False) = x)

-- Write E x for (R x -> False).
-- K1: if x is a value of E, then E x is refuted.
theorem K1 (B x : Prop) (h : (R B -> False) = x) : Not (R x -> False) :=
  fun e => e (Eq.mp (congrArg R h) (mkr B))

-- K3: if x is not a value of E, then E x holds.
theorem K3 (x : Prop) (hn : Not (Exists (fun B : Prop => (R B -> False) = x))) :
    R x -> False :=
  fun r => hn (R_inv x r)

theorem boom : False := by
  -- Classical.propComplete (a : Prop) : a = True \/ a = False
  have ha := Classical.propComplete (R False -> False)
  have hb := Classical.propComplete (R True -> False)
  cases ha with
  | inl ha =>
    -- E False = True: True is a value of E, so E True is refuted, so E True = False:
    -- False is a value of E, so E False is refuted
    have n1 : Not (R True -> False) := K1 False True ha
    have hb' : (R True -> False) = False :=
      propext (Iff.intro (fun h => n1 h) (fun h => False.elim h))
    have n0 : Not (R False -> False) := K1 True False hb'
    exact n0 (Eq.mpr ha True.intro)
  | inr ha =>
    cases hb with
    | inl hb =>
      -- E True = True: True is a value of E, so E True is refuted
      have n1 : Not (R True -> False) := K1 True True hb
      exact n1 (Eq.mpr hb True.intro)
    | inr hb =>
      -- E False = False and E True = False: True is not a value of E, so E True holds
      have hni : Not (Exists (fun B : Prop => (R B -> False) = True)) := by
        intro hex
        cases hex with
        | intro B hB =>
          cases Classical.propComplete B with
          | inl hBT =>
            have e : (R B -> False) = (R True -> False) :=
              congrArg (fun P => R P -> False) hBT
            exact Eq.mpr (Eq.trans (Eq.symm (Eq.trans e hb)) hB) True.intro
          | inr hBF =>
            have e : (R B -> False) = (R False -> False) :=
              congrArg (fun P => R P -> False) hBF
            exact Eq.mpr (Eq.trans (Eq.symm (Eq.trans e ha)) hB) True.intro
      exact Eq.mp hb (K3 True hni)

#print axioms boom
