import read_iris
import svm1
import cvxpy

data, labels = read_iris.readBinaryClasses('./iris.csv')
#print data

a, b = svm1.createVars(data.shape[1])
constraints = svm1.createConstraints(a, b, data, labels)

obj = cvxpy.Minimize(cvxpy.norm2(a))
prob = cvxpy.Problem(obj, constraints)
print prob.solve()
print 'A'
print a.value
print 'B', b.value

svm1.test(a.value, b.value, data, labels)
