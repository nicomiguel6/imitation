import casadi as ca

x = ca.MX.sym('x')
f = (x - 3)**2
nlp = {'x': x, 'f': f}

solver = ca.nlpsol('s', 'ipopt', nlp, {
    'ipopt': {
        'linear_solver': 'ma57',   # or 'ma27', 'ma77', 'ma97'
        'print_level': 5
    }
})

sol = solver(x0=0)
print("✅ Optimal x:", float(sol['x']))
