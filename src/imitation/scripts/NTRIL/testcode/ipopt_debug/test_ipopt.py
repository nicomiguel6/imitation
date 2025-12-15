import casadi as ca

x = ca.MX.sym("x")
nlp = {"x": x, "f": (x - 3) ** 2}

opts = {
    "ipopt.linear_solver": "ma57",
    "ipopt.hsllib": "/usr/local/lib/libcoinhsl.dylib",
    "ipopt.print_level": 5,
}

S = ca.nlpsol("S", "ipopt", nlp, opts)
sol = S(x0=0)
print("x* =", float(sol["x"]))
