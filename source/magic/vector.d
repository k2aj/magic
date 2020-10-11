module magic.vector;

import std.math : approxEqual, PI, sqrt;
import std.range;
import std.algorithm;
import std.format : format;
import std.conv : text, to;
import std.traits : isFloatingPoint, isIntegral;
import std.meta : Repeat, allSatisfy, Alias, staticMap;
import std.traits : Unconst;

unittest {
	//v.array, v.e and xyzw,rgba should have the same data
	immutable v = Vector!(int,4)(1,2,3,4);
	assert(v.array == [1,2,3,4]);
	assert([v.e] == v.array);
	assert(v.x == v.r && v.r == v[0]);
	assert(v.y == v.g && v.g == v[1]);
	assert(v.z == v.b && v.b == v[2]);
	assert(v.w == v.a && v.a == v[3]);
	
	//Vector factory functions
	assert(vector(1,2,3,4,5) == [1,2,3,4,5]);
	assert(vec2.polar(PI/4).cwise!approxEqual(vec2(1,1).normalize)[].all);
	assert(vec2.polar(0,1337).cwise!approxEqual(vec2(1337,0))[].all);
	
	//Expand operation
	assert([v.e, v.e, v.e] == iota(1,5).cycle.take(12).array);
	assert([v.e[1..3]] == [2,3]);
	
	//Swizzling operations
	assert(v.xyzw == v.rgba && v.rgba == v);
	assert(v.zyz == [3,2,3]);
	assert(v.wwwww == [4,4,4,4,4]);
	assert(v.bgra == [3,2,1,4]);
	assert(v.xyxzwwxy == [1,2,1,3,4,4,1,2]);
	//Invalid swizzle mask should not compile
	static assert(!__traits(compiles, v.foobar));
	//Swizzling components not present in vector should not compile
	static assert(!__traits(compiles, v.xyz.www));

	//Test assignment operators
	ivec4 u;
	u = 2;
	assert(u[].all!(x => x==2));
	u = [1,2,3,4];
	assert(u == [1,2,3,4]);
}

unittest {
	//Arithmetic operators should work component-wise 
	//v op u == [v.x op u.x, v.y op u.y, ...]
	immutable ivec3 v = [1,2,3], u = [4,5,6];
	assert(v + u == [5,7,9]);
	assert(v + [4,5,6] == [5,7,9]);
	assert(v - u == ivec3(-3));
	assert(v * u == [4,10,18]);
	assert(u / v == [4,2,2]);
	assert(u % v == [0,1,0]);
	
	//Vector op scalar treats scalar like a vector filled with that value
	assert(v + 3 == u);
	assert(3 + v == u);
	assert(u / 2 == ivec3(2,2,3));
	
	//opOpAssign operations, taking both vectors & scalars as arguments
	ivec3 w = [7,8,9];
	w += v;
	assert(w == [8,10,12]);
	w -= u;
	assert(w == u);
	w *= v;
	assert(w == v*u);
	w /= u;
	assert(w == v);
	w += 5;
	assert(w == v+5);
	w *= 7;
	assert(w == (v+5)*7);
	w /= 3;
	assert(w == ((v+5)*7)/3);
}
unittest {
	//Dot and cross product
	assert(dot(ivec3(1,2,3), ivec3(4,5,6)) == 32);
	assert(cross(ivec3(1,2,3), ivec3(4,5,6)) == [-3,6,-3]);

	//norm(v) returns 2-norm (magnitude) of the vector
	import std.math : approxEqual;
	assert(approxEqual(vec2(3,4).norm, 5));
	assert(approxEqual(vec3(3,4,12).norm, 13)); 

	//normalize(v) makes the norm 1, while preserving the ratio between vector components
	assert(vec2(3,4).normalize.cwise!approxEqual(vec2(0.6f,0.8f))[].all);
	assert(vec2(3,4).normalizeOrZero.cwise!approxEqual(vec2(3,4).normalize)[].all);
	assert(vec2(0).normalizeOrZero == vec2(0));

	//scaleTo(v,n) works like normalize, except it makes the norm equal to n
	assert(vec2(3,4).scaleTo(3).cwise!approxEqual(vec2(1.8f, 2.4f))[].all);
}
unittest {
	/*cwise makes scalar-valued functions operate on vector arguments
	  component-wise and return a vector of results*/
	import std.math : abs, approxEqual;
	assert(cwise!max(vec3(1,2,3), vec3(3,2,1)) == [3,2,3]);
	assert(cwise!max(vec4(1,2,3,4), vec4(3)) == [3,3,3,4]); 
	assert(cwise!abs(vector(-3,-2,-1,0,1,2)) == [3,2,1,0,1,2]);
	assert(ivec4(1,2,3,4).cwise!clamp(ivec4(0,3,1,2),ivec4(2,4,2,6)) == ivec4(1,3,2,4));
	
	ivec3 v = [1,2,3];
	assert(cwise!approxEqual(v, v.to!vec3)[].all);
}

/++ Vector data type, inspired by GLSL vectors.
	Examples:
	---
	Vector!(int,3) v = [1,2,3]; //Create from static array
	ivec3 u = ivec3(4,5,6); 
	ivec4 w = ivec4(2); //Fills all components with given value
	auto k = vector(1,2,5,-3,7); //Create using vector() factory function
	
	//Arithmetic operators work component-wise
	assert(v*u == [4,10,18]); 
	
	assert(vec2(3,4).norm.approxEqual(5f)); 
	assert(dot(ivec2(2,3), ivec2(4,5)) == 23);
	
	//cwise allows scalar-accepting functions to work on vectors!
	assert(k.cwise!abs == [1,2,5,3,7]); 
	//Works with functions of any arity! (probably)
	assert(cwise!max(vec3(1,2,3), vec3(3,2,1)) == [3,2,3]);
	
	//Can be expanded like a tuple too!
	void foo(int r, int g, int b, int a);
	foo(w.expand);
	foo(w.e);
	---
+/
pure nothrow @safe @nogc struct Vector(T,size_t _length)
{
	/// How many components this `Vector` has
	enum length = _length;
	///
	alias ComponentType = T;
	
	union 
	{
		/** This `Vector`'s components stored as static array. */
		T[length] array;
		struct 
		{
			/++ Allows `Vector` to be expanded like `std.typecons.Tuple`.
				Examples:
				---
				vec3 getBackgroundColor() {/*code for calculating background color*/}
				import bindbc.opengl;
				glClearColor(getBackgroundColor().expand, 1f);
				---
				vec3 v;
				vec4 u = [v.e, 1f];
				---
			+/
			Repeat!(length,T) expand; 
			alias e = expand;
			
			// Generate named fields: x,y,z,w, r,g,b,a
			static foreach(index,names; vectorComponents)
				static if(index < length)
					static foreach(name; names)
						mixin(text("alias ",name," = e[",index,"];"));
		}
	}
	
	alias array this;
	
	/// Creates a `Vector` from a static array of components.
	this(T[length] components...){array = components;}
	/// Creates a `Vector` with all components set to given value.
	this(T filler){array[] = filler;}
	
	static if(length == 2 && isFloatingPoint!T)
	{
		/** Params:
				angle = Angle between the created `Vector` and unit vector [1,0], in radians.
				norm = 
			Returns: 2D `Vector` constructed from polar coordinates.
		*/
		static Vector polar(T angle, T norm=1) {
			import std.math : sin,cos;
			return Vector(cos(angle)*norm, sin(angle)*norm);
		}
	}
	
	/// Sets all components to values from given static array.
	void opAssign(T[length] components){array = components;}
	/// Sets all components to given value.
	void opAssign(T filler){array[] = filler;}
	
	Vector!(T,length) opUnary(string op)() const 
	{
		mixin("T[length] result = "~op~"array[];");
		return Vector!(T,length)(result);
	}
	Vector!(T,length) opBinary(string op)(auto ref const T[length] rhs) const 
	{
		T[length] result;
		mixin("result[] = array[]"~op~"rhs[];");
		return Vector!(T,length)(result);
	}
	auto opBinary(string op,S)(const S rhs) const 
	if(!isVector!S && is(S : T) && op != "in") 
	{
		mixin("T[length] result = array[]"~op~"rhs;");
		return Vector!(T,length)(result);
	}
	auto opBinaryRight(string op,S)(const S lhs) const 
	if(!isVector!S && is(S : T)) 
	{
		mixin("T[length] result = lhs"~op~"array[];");
		return Vector!(T,length)(result);
	}
	void opOpAssign(string op)(auto ref const T[length] rhs) 
	{
		mixin("array[]"~op~"= rhs[];");
	}
	void opOpAssign(string op,S)(const S rhs) 
	if(!isVector!S && is(S : T)) 
	{
		mixin("array[]"~op~"= rhs;");
	}
	V opCast(V)() const 
	if(isVector!V && V.length == length) 
	{
		alias C = Unconst!(V.ComponentType);
		C[length] arr;
		static foreach(i; 0..length) 
			arr[i] = cast(C) array[i];
		return V(arr);
	}
	
	/// Swizzlling
	auto opDispatch(string swizzleMask)() const 
	if(swizzleMask.length >= 2) 
	{
		static foreach(comp; swizzleMask)
			static if(!__traits(compiles, mixin(comp.to!string)))
				static assert(0, "Invalid character '%s' found in swizzle mask \"%s\"".format(comp,swizzleMask));
		mixin("return vector("~
			swizzleMask.slide(1).map!(to!string).join(",")
		~");");
	}
	
	/// Returns: Pointer to the first component of this `Vector`.
	@property inout(T*) ptr() inout @trusted 
	{
		return array.ptr;
	}
}

/// Returns: `Vector` with given components.
Vector!(T,length) vector(T,size_t length)(const T[length] components...) pure nothrow @safe @nogc 
{
	return Vector!(T,length)(components);
}

/// Returns: Dot product of the arguments.
T dot(T,size_t length)(auto ref const T[length] v, auto ref const T[length] u) pure nothrow @safe @nogc 
{
	T result = 0;
	foreach(i; 0..length)
		result += v[i]*u[i];
	return result;
}
///Returns: Cross product of the arguments.
Vector!(T,3) cross(T)(auto ref const T[3] v, auto ref const T[3] u) pure nothrow @safe @nogc 
{
	return Vector!(T,3)(v[1]*u[2] - v[2]*u[1], v[2]*u[0] - v[0]*u[2], v[0]*u[1] - v[1]*u[0]);
}
/// Returns: 2-norm (also known as length/magnitude) of given vector.
T norm(T,size_t length)(auto ref const T[length] v) pure nothrow @safe @nogc 
if(isFloatingPoint!T) 
{
	return sqrt(dot(v,v));
}
/// Returns: 2-norm (also known as length/magnitude) of given vector.
double norm(T,size_t length)(auto ref const T[length] v) pure nothrow @safe @nogc 
if(isIntegral!T) 
{
	return sqrt(cast(double) dot(v,v));
}
/// Returns: `Vector` with the same direction as `v`, but norm equal to 1.
auto normalize(T,size_t size)(auto ref const T[size] v) pure nothrow @safe @nogc 
if(isFloatingPoint!T) in(norm(v) != 0) 
{
	return 1/norm(v) * vector(v);
}
/// Returns: `normalize(v)` if `v` has non-zero norm, otherwise returns a vector with all components set to 0
auto normalizeOrZero(T,size_t size)(auto ref const T[size] v)
{
	const norm2 = dot(v,v);
	static if(isFloatingPoint!T) return norm2.approxEqual(0) ? Vector!(T,size)(0) : 1/sqrt(norm2) * vector(v);
	else return norm2 == 0 ? Vector!(double,size)(0) : 1/sqrt(cast(double) norm2) * vector(v);
}
/// Returns: `normalize(v)` if `v` has non-zero norm, otherwise returns a vector with x=1 and all other components set to 0
auto normalizeOrUnitX(T,size_t size)(auto ref const T[size] v)
{
	const norm2 = dot(v,v);
	static if(isFloatingPoint!T) return norm2.approxEqual(0) ? Vector!(T,size)(1,Repeat!(size-1,0)) : 1/sqrt(norm2) * vector(v);
	else return norm2 == 0 ? Vector!(double,size)(1,Repeat!(size-1,0)) : 1/sqrt(cast(double) norm2) * vector(v);
}

/// Returns: `Vector` with the same direction as `v`, but norm equal to `targetNorm`.
auto scaleTo(T,size_t size)(auto ref const T[size] v, T targetNorm) pure nothrow @safe @nogc 
if(isFloatingPoint!T) 
{
	return (targetNorm/norm(v)) * vector(v);
}

/++ Given a function `op` which operates on scalars, applies it to `Vector` arguments component-wise.
	At least one argument must be a `Vector`. All `Vector` arguments must have the same length.
	---
	//Equivalent to vec3(max(1,3), max(2,2), max(3,1))
	assert(cwise!max(vec3(1,2,3), vec3(3,2,1)) == vec3(3,2,3));
	---
++/
auto cwise(alias op, Args...)(auto ref Args args) pure nothrow @safe @nogc if(
	allSatisfy!(isVector, Args) && 
	allSatisfy!(Equal!(Alias!(Args[0].length)), staticMap!(VectorLength,Args))
) {
	//We don't return anything if op returns void
	static if(is(typeof(op(mixin(cwiseCall(Args.length, 0)))) == void))
		static foreach(i; 0..Args[0].length) 
			mixin(cwiseCall(Args.length, i));
	else mixin("return vector("~iota(Args[0].length).map!(i => cwiseCall(Args.length,i)).join(",")~");");
}
///
enum isVector(T) = is(T == Vector!(U,length), U, size_t length);
///
template VectorLength(T) 
if(isVector!T) 
{
	enum VectorLength = T.length;
}
///
template VectorComponentType(T) 
if(isVector!T) 
{
	alias VectorComponentType = T.ComponentType;
}

/* Short aliases for commonly used vector types */
alias vec(size_t length) = Vector!(float,length);
alias dvec(size_t length) = Vector!(double,length);
alias rvec(size_t length) = Vector!(real,length);
alias ivec(size_t length) = Vector!(int,length);
alias uvec(size_t length) = Vector!(uint,length);
alias usvec(size_t length) = Vector!(ushort,length);
alias ubvec(size_t length) = Vector!(ubyte,length);

alias vec2 = vec!2;
alias vec3 = vec!3;
alias vec4 = vec!4;

alias dvec2 = dvec!2;
alias dvec3 = dvec!3;
alias dvec4 = dvec!4;

alias rvec2 = rvec!2;
alias rvec3 = rvec!3;
alias rvec4 = rvec!4;

alias ivec2 = ivec!2;
alias ivec3 = ivec!3;
alias ivec4 = ivec!4;

alias uvec2 = uvec!2;
alias uvec3 = uvec!3;
alias uvec4 = uvec!4;

alias ubvec2 = ubvec!2;
alias ubvec3 = ubvec!3;
alias ubvec4 = ubvec!4;

alias usvec2 = usvec!2;
alias usvec3 = usvec!3;
alias usvec4 = usvec!4;

private:
//Creates mixin string for function call `op(args[0][i], args[1][i], args[2][i], ...)`
string cwiseCall(size_t argsLength, size_t index)
{
	return "op(" ~ iota(argsLength).map!(i => "args[%s][%s]".format(i,index)).join(", ") ~ ")";
}
enum vectorComponents = [
	["x","r"],
	["y","g"],
	["z","b"],
	["w","a"]
];

template Equal(alias val){
	template Equal(alias x){
		enum Equal = x == val;
	}
}