function op_Plus(x, y){
        return x + y
};

function op_Minus(x, y){
        return x - y
};

function op_Mul(x, y){
        return x * y
};

function op_Div(x, y){
        return x / y
};

function print (x){ console.log(x) ; };


var op_Dot = (function (f,x){ return f(x);});
 
var g = (function (a){ return op_Mul(2,a);});
 
var f = (function (b){ return op_Plus(10, b);});
 
var main = (function (){ return print(op_Dot(g, f(10)));});
 

main()
