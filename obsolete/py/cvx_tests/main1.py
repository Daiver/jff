
from cvxpy import *

# Create two scalar optimization variables.
x = Variable()
y = Variable()

# Create two constraints.
constraints = [x + y == 1,
               -x + y >= 1]

# Form objective.
#obj = Minimize(square(x - y))
obj = Minimize(square(x))

# Form and solve problem.
prob = Problem(obj, constraints)
prob.solve()

# The optimal dual variable (Lagrange multiplier) for
# a constraint is stored in constraint.dual_value.
print "optimal (x + y == 1) dual variable", constraints[0].dual_value
print "optimal (x - y >= 1) dual variable", constraints[1].dual_value
print "x - y value:", (x - y).value
print x.value
print y.value
