nn(net_baseline,[X],Y,[sevbrady,brady,normal,tachy]) :: baseline(X,Y).
nn(net_variab,[X],Y,[reduced,normal,increased,sinusoidal]) :: variability(X,Y).
nn(net_accel,[X]) :: accel(X).
nn(net_rep_any,[X]) :: rep_any(X).
nn(net_rep_late,[X]) :: rep_late(X).
nn(net_rep_variable,[X]) :: rep_variable(X).
nn(net_early,[X]) :: early(X).
nn(net_prolonged,[X]) :: prolonged(X).
nn(net_very_prolonged,[X]) :: very_prolonged(X).
nn(net_reduced_var_long,[X]) :: reduced_var_long(X).
nn(net_increased_var_long,[X]) :: increased_var_long(X).
nn(net_rep_late_long,[X]) :: rep_late_long(X).
pathological(X) :- baseline(X,sevbrady).
pathological(X) :- reduced_var_long(X).
pathological(X) :- increased_var_long(X).
pathological(X) :- variability(X,sinusoidal).
pathological(X) :- rep_late_long(X).
pathological(X) :- very_prolonged(X).
normal(X) :- baseline(X,normal), variability(X,normal), \+rep_any(X), \+prolonged(X), \+pathological(X).
suspicious(X) :- \+pathological(X), \+normal(X).
t(0.05) :: acid(X).
t(0.1) :: acid(X) :- pathological(X).
t(0.1) :: acid(X) :- suspicious(X).
t(0.1) :: acid(X) :- rep_late(X).
t(0.1) :: acid(X) :- rep_variable(X).
t(0.1) :: acid(X) :- prolonged(X).
t(0.1) :: acid(X) :- baseline(X,tachy).
t(0.1) :: acid(X) :- \+accel(X).
t(0.1) :: acid(X) :- variability(X,reduced).
t(0.1) :: acid(X) :- rep_any(X).