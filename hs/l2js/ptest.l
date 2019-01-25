
(.) = (f x) -> (f x);
g = (a) -> ((*) 2 a);
f = (b) -> 10 + b;
main = () -> (print (g . f 10))
