��

��
D
AddV2
x"T
y"T
z"T"
Ttype:
2	��
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( �
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
$
DisableCopyOnRead
resource�
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
.
Identity

input"T
output"T"	
Ttype
�
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:
2	"
grad_abool( "
grad_bbool( 
�
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( �
?
Mul
x"T
y"T
z"T"
Ttype:
2	�

NoOp
U
NotEqual
x"T
y"T
z
"	
Ttype"$
incompatible_shape_errorbool(�
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
�
ParseExampleV2

serialized	
names
sparse_keys

dense_keys
ragged_keys
dense_defaults2Tdense
sparse_indices	*
num_sparse
sparse_values2sparse_types
sparse_shapes	*
num_sparse
dense_values2Tdense#
ragged_values2ragged_value_types'
ragged_row_splits2ragged_split_types"
Tdense
list(type)(:
2	"

num_sparseint("%
sparse_types
list(type)(:
2	"+
ragged_value_types
list(type)(:
2	"*
ragged_split_types
list(type)(:
2	"
dense_shapeslist(shape)(
�
PartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
X
PlaceholderWithDefault
input"dtype
output"dtype"
dtypetype"
shapeshape
@
ReadVariableOp
resource
value"dtype"
dtypetype�
@
RealDiv
x"T
y"T
z"T"
Ttype:
2	
E
Relu
features"T
activations"T"
Ttype:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
.
Rsqrt
x"T
y"T"
Ttype:

2
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
?
Select
	condition

t"T
e"T
output"T"	
Ttype
A
SelectV2
	condition

t"T
e"T
output"T"	
Ttype
d
Shape

input"T&
output"out_type��out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
9
Softmax
logits"T
softmax"T"
Ttype:
2
-
Sqrt
x"T
y"T"
Ttype:

2
�
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ��
@
StaticRegexFullMatch	
input

output
"
patternstring
�
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
L

StringJoin
inputs*N

output"

Nint("
	separatorstring 
<
Sub
x"T
y"T
z"T"
Ttype:
2	
�
VarHandleOp
resource"
	containerstring "
shared_namestring "

debug_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �
9
VarIsInitializedOp
resource
is_initialized
�
&
	ZerosLike
x"T
y"T"	
Ttype"serve*2.16.22v2.16.1-19-g810f233968c8��	
J
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *hKB
L
Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *ʝ�A
L
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *7*�C
L
Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *�FB
�
adam/dense_2_bias_velocityVarHandleOp*
_output_shapes
: *+

debug_nameadam/dense_2_bias_velocity/*
dtype0*
shape:*+
shared_nameadam/dense_2_bias_velocity
�
.adam/dense_2_bias_velocity/Read/ReadVariableOpReadVariableOpadam/dense_2_bias_velocity*
_output_shapes
:*
dtype0
�
#Variable/Initializer/ReadVariableOpReadVariableOpadam/dense_2_bias_velocity*
_class
loc:@Variable*
_output_shapes
:*
dtype0
�
VariableVarHandleOp*
_class
loc:@Variable*
_output_shapes
: *

debug_name	Variable/*
dtype0*
shape:*
shared_name
Variable
a
)Variable/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable*
_output_shapes
: 
_
Variable/AssignAssignVariableOpVariable#Variable/Initializer/ReadVariableOp*
dtype0
a
Variable/Read/ReadVariableOpReadVariableOpVariable*
_output_shapes
:*
dtype0
�
adam/dense_2_bias_momentumVarHandleOp*
_output_shapes
: *+

debug_nameadam/dense_2_bias_momentum/*
dtype0*
shape:*+
shared_nameadam/dense_2_bias_momentum
�
.adam/dense_2_bias_momentum/Read/ReadVariableOpReadVariableOpadam/dense_2_bias_momentum*
_output_shapes
:*
dtype0
�
%Variable_1/Initializer/ReadVariableOpReadVariableOpadam/dense_2_bias_momentum*
_class
loc:@Variable_1*
_output_shapes
:*
dtype0
�

Variable_1VarHandleOp*
_class
loc:@Variable_1*
_output_shapes
: *

debug_nameVariable_1/*
dtype0*
shape:*
shared_name
Variable_1
e
+Variable_1/IsInitialized/VarIsInitializedOpVarIsInitializedOp
Variable_1*
_output_shapes
: 
e
Variable_1/AssignAssignVariableOp
Variable_1%Variable_1/Initializer/ReadVariableOp*
dtype0
e
Variable_1/Read/ReadVariableOpReadVariableOp
Variable_1*
_output_shapes
:*
dtype0
�
adam/dense_2_kernel_velocityVarHandleOp*
_output_shapes
: *-

debug_nameadam/dense_2_kernel_velocity/*
dtype0*
shape
:@*-
shared_nameadam/dense_2_kernel_velocity
�
0adam/dense_2_kernel_velocity/Read/ReadVariableOpReadVariableOpadam/dense_2_kernel_velocity*
_output_shapes

:@*
dtype0
�
%Variable_2/Initializer/ReadVariableOpReadVariableOpadam/dense_2_kernel_velocity*
_class
loc:@Variable_2*
_output_shapes

:@*
dtype0
�

Variable_2VarHandleOp*
_class
loc:@Variable_2*
_output_shapes
: *

debug_nameVariable_2/*
dtype0*
shape
:@*
shared_name
Variable_2
e
+Variable_2/IsInitialized/VarIsInitializedOpVarIsInitializedOp
Variable_2*
_output_shapes
: 
e
Variable_2/AssignAssignVariableOp
Variable_2%Variable_2/Initializer/ReadVariableOp*
dtype0
i
Variable_2/Read/ReadVariableOpReadVariableOp
Variable_2*
_output_shapes

:@*
dtype0
�
adam/dense_2_kernel_momentumVarHandleOp*
_output_shapes
: *-

debug_nameadam/dense_2_kernel_momentum/*
dtype0*
shape
:@*-
shared_nameadam/dense_2_kernel_momentum
�
0adam/dense_2_kernel_momentum/Read/ReadVariableOpReadVariableOpadam/dense_2_kernel_momentum*
_output_shapes

:@*
dtype0
�
%Variable_3/Initializer/ReadVariableOpReadVariableOpadam/dense_2_kernel_momentum*
_class
loc:@Variable_3*
_output_shapes

:@*
dtype0
�

Variable_3VarHandleOp*
_class
loc:@Variable_3*
_output_shapes
: *

debug_nameVariable_3/*
dtype0*
shape
:@*
shared_name
Variable_3
e
+Variable_3/IsInitialized/VarIsInitializedOpVarIsInitializedOp
Variable_3*
_output_shapes
: 
e
Variable_3/AssignAssignVariableOp
Variable_3%Variable_3/Initializer/ReadVariableOp*
dtype0
i
Variable_3/Read/ReadVariableOpReadVariableOp
Variable_3*
_output_shapes

:@*
dtype0
�
(adam/batch_normalization_1_beta_velocityVarHandleOp*
_output_shapes
: *9

debug_name+)adam/batch_normalization_1_beta_velocity/*
dtype0*
shape:@*9
shared_name*(adam/batch_normalization_1_beta_velocity
�
<adam/batch_normalization_1_beta_velocity/Read/ReadVariableOpReadVariableOp(adam/batch_normalization_1_beta_velocity*
_output_shapes
:@*
dtype0
�
%Variable_4/Initializer/ReadVariableOpReadVariableOp(adam/batch_normalization_1_beta_velocity*
_class
loc:@Variable_4*
_output_shapes
:@*
dtype0
�

Variable_4VarHandleOp*
_class
loc:@Variable_4*
_output_shapes
: *

debug_nameVariable_4/*
dtype0*
shape:@*
shared_name
Variable_4
e
+Variable_4/IsInitialized/VarIsInitializedOpVarIsInitializedOp
Variable_4*
_output_shapes
: 
e
Variable_4/AssignAssignVariableOp
Variable_4%Variable_4/Initializer/ReadVariableOp*
dtype0
e
Variable_4/Read/ReadVariableOpReadVariableOp
Variable_4*
_output_shapes
:@*
dtype0
�
(adam/batch_normalization_1_beta_momentumVarHandleOp*
_output_shapes
: *9

debug_name+)adam/batch_normalization_1_beta_momentum/*
dtype0*
shape:@*9
shared_name*(adam/batch_normalization_1_beta_momentum
�
<adam/batch_normalization_1_beta_momentum/Read/ReadVariableOpReadVariableOp(adam/batch_normalization_1_beta_momentum*
_output_shapes
:@*
dtype0
�
%Variable_5/Initializer/ReadVariableOpReadVariableOp(adam/batch_normalization_1_beta_momentum*
_class
loc:@Variable_5*
_output_shapes
:@*
dtype0
�

Variable_5VarHandleOp*
_class
loc:@Variable_5*
_output_shapes
: *

debug_nameVariable_5/*
dtype0*
shape:@*
shared_name
Variable_5
e
+Variable_5/IsInitialized/VarIsInitializedOpVarIsInitializedOp
Variable_5*
_output_shapes
: 
e
Variable_5/AssignAssignVariableOp
Variable_5%Variable_5/Initializer/ReadVariableOp*
dtype0
e
Variable_5/Read/ReadVariableOpReadVariableOp
Variable_5*
_output_shapes
:@*
dtype0
�
)adam/batch_normalization_1_gamma_velocityVarHandleOp*
_output_shapes
: *:

debug_name,*adam/batch_normalization_1_gamma_velocity/*
dtype0*
shape:@*:
shared_name+)adam/batch_normalization_1_gamma_velocity
�
=adam/batch_normalization_1_gamma_velocity/Read/ReadVariableOpReadVariableOp)adam/batch_normalization_1_gamma_velocity*
_output_shapes
:@*
dtype0
�
%Variable_6/Initializer/ReadVariableOpReadVariableOp)adam/batch_normalization_1_gamma_velocity*
_class
loc:@Variable_6*
_output_shapes
:@*
dtype0
�

Variable_6VarHandleOp*
_class
loc:@Variable_6*
_output_shapes
: *

debug_nameVariable_6/*
dtype0*
shape:@*
shared_name
Variable_6
e
+Variable_6/IsInitialized/VarIsInitializedOpVarIsInitializedOp
Variable_6*
_output_shapes
: 
e
Variable_6/AssignAssignVariableOp
Variable_6%Variable_6/Initializer/ReadVariableOp*
dtype0
e
Variable_6/Read/ReadVariableOpReadVariableOp
Variable_6*
_output_shapes
:@*
dtype0
�
)adam/batch_normalization_1_gamma_momentumVarHandleOp*
_output_shapes
: *:

debug_name,*adam/batch_normalization_1_gamma_momentum/*
dtype0*
shape:@*:
shared_name+)adam/batch_normalization_1_gamma_momentum
�
=adam/batch_normalization_1_gamma_momentum/Read/ReadVariableOpReadVariableOp)adam/batch_normalization_1_gamma_momentum*
_output_shapes
:@*
dtype0
�
%Variable_7/Initializer/ReadVariableOpReadVariableOp)adam/batch_normalization_1_gamma_momentum*
_class
loc:@Variable_7*
_output_shapes
:@*
dtype0
�

Variable_7VarHandleOp*
_class
loc:@Variable_7*
_output_shapes
: *

debug_nameVariable_7/*
dtype0*
shape:@*
shared_name
Variable_7
e
+Variable_7/IsInitialized/VarIsInitializedOpVarIsInitializedOp
Variable_7*
_output_shapes
: 
e
Variable_7/AssignAssignVariableOp
Variable_7%Variable_7/Initializer/ReadVariableOp*
dtype0
e
Variable_7/Read/ReadVariableOpReadVariableOp
Variable_7*
_output_shapes
:@*
dtype0
�
adam/dense_1_bias_velocityVarHandleOp*
_output_shapes
: *+

debug_nameadam/dense_1_bias_velocity/*
dtype0*
shape:@*+
shared_nameadam/dense_1_bias_velocity
�
.adam/dense_1_bias_velocity/Read/ReadVariableOpReadVariableOpadam/dense_1_bias_velocity*
_output_shapes
:@*
dtype0
�
%Variable_8/Initializer/ReadVariableOpReadVariableOpadam/dense_1_bias_velocity*
_class
loc:@Variable_8*
_output_shapes
:@*
dtype0
�

Variable_8VarHandleOp*
_class
loc:@Variable_8*
_output_shapes
: *

debug_nameVariable_8/*
dtype0*
shape:@*
shared_name
Variable_8
e
+Variable_8/IsInitialized/VarIsInitializedOpVarIsInitializedOp
Variable_8*
_output_shapes
: 
e
Variable_8/AssignAssignVariableOp
Variable_8%Variable_8/Initializer/ReadVariableOp*
dtype0
e
Variable_8/Read/ReadVariableOpReadVariableOp
Variable_8*
_output_shapes
:@*
dtype0
�
adam/dense_1_bias_momentumVarHandleOp*
_output_shapes
: *+

debug_nameadam/dense_1_bias_momentum/*
dtype0*
shape:@*+
shared_nameadam/dense_1_bias_momentum
�
.adam/dense_1_bias_momentum/Read/ReadVariableOpReadVariableOpadam/dense_1_bias_momentum*
_output_shapes
:@*
dtype0
�
%Variable_9/Initializer/ReadVariableOpReadVariableOpadam/dense_1_bias_momentum*
_class
loc:@Variable_9*
_output_shapes
:@*
dtype0
�

Variable_9VarHandleOp*
_class
loc:@Variable_9*
_output_shapes
: *

debug_nameVariable_9/*
dtype0*
shape:@*
shared_name
Variable_9
e
+Variable_9/IsInitialized/VarIsInitializedOpVarIsInitializedOp
Variable_9*
_output_shapes
: 
e
Variable_9/AssignAssignVariableOp
Variable_9%Variable_9/Initializer/ReadVariableOp*
dtype0
e
Variable_9/Read/ReadVariableOpReadVariableOp
Variable_9*
_output_shapes
:@*
dtype0
�
adam/dense_1_kernel_velocityVarHandleOp*
_output_shapes
: *-

debug_nameadam/dense_1_kernel_velocity/*
dtype0*
shape:	�@*-
shared_nameadam/dense_1_kernel_velocity
�
0adam/dense_1_kernel_velocity/Read/ReadVariableOpReadVariableOpadam/dense_1_kernel_velocity*
_output_shapes
:	�@*
dtype0
�
&Variable_10/Initializer/ReadVariableOpReadVariableOpadam/dense_1_kernel_velocity*
_class
loc:@Variable_10*
_output_shapes
:	�@*
dtype0
�
Variable_10VarHandleOp*
_class
loc:@Variable_10*
_output_shapes
: *

debug_nameVariable_10/*
dtype0*
shape:	�@*
shared_nameVariable_10
g
,Variable_10/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_10*
_output_shapes
: 
h
Variable_10/AssignAssignVariableOpVariable_10&Variable_10/Initializer/ReadVariableOp*
dtype0
l
Variable_10/Read/ReadVariableOpReadVariableOpVariable_10*
_output_shapes
:	�@*
dtype0
�
adam/dense_1_kernel_momentumVarHandleOp*
_output_shapes
: *-

debug_nameadam/dense_1_kernel_momentum/*
dtype0*
shape:	�@*-
shared_nameadam/dense_1_kernel_momentum
�
0adam/dense_1_kernel_momentum/Read/ReadVariableOpReadVariableOpadam/dense_1_kernel_momentum*
_output_shapes
:	�@*
dtype0
�
&Variable_11/Initializer/ReadVariableOpReadVariableOpadam/dense_1_kernel_momentum*
_class
loc:@Variable_11*
_output_shapes
:	�@*
dtype0
�
Variable_11VarHandleOp*
_class
loc:@Variable_11*
_output_shapes
: *

debug_nameVariable_11/*
dtype0*
shape:	�@*
shared_nameVariable_11
g
,Variable_11/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_11*
_output_shapes
: 
h
Variable_11/AssignAssignVariableOpVariable_11&Variable_11/Initializer/ReadVariableOp*
dtype0
l
Variable_11/Read/ReadVariableOpReadVariableOpVariable_11*
_output_shapes
:	�@*
dtype0
�
&adam/batch_normalization_beta_velocityVarHandleOp*
_output_shapes
: *7

debug_name)'adam/batch_normalization_beta_velocity/*
dtype0*
shape:�*7
shared_name(&adam/batch_normalization_beta_velocity
�
:adam/batch_normalization_beta_velocity/Read/ReadVariableOpReadVariableOp&adam/batch_normalization_beta_velocity*
_output_shapes	
:�*
dtype0
�
&Variable_12/Initializer/ReadVariableOpReadVariableOp&adam/batch_normalization_beta_velocity*
_class
loc:@Variable_12*
_output_shapes	
:�*
dtype0
�
Variable_12VarHandleOp*
_class
loc:@Variable_12*
_output_shapes
: *

debug_nameVariable_12/*
dtype0*
shape:�*
shared_nameVariable_12
g
,Variable_12/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_12*
_output_shapes
: 
h
Variable_12/AssignAssignVariableOpVariable_12&Variable_12/Initializer/ReadVariableOp*
dtype0
h
Variable_12/Read/ReadVariableOpReadVariableOpVariable_12*
_output_shapes	
:�*
dtype0
�
&adam/batch_normalization_beta_momentumVarHandleOp*
_output_shapes
: *7

debug_name)'adam/batch_normalization_beta_momentum/*
dtype0*
shape:�*7
shared_name(&adam/batch_normalization_beta_momentum
�
:adam/batch_normalization_beta_momentum/Read/ReadVariableOpReadVariableOp&adam/batch_normalization_beta_momentum*
_output_shapes	
:�*
dtype0
�
&Variable_13/Initializer/ReadVariableOpReadVariableOp&adam/batch_normalization_beta_momentum*
_class
loc:@Variable_13*
_output_shapes	
:�*
dtype0
�
Variable_13VarHandleOp*
_class
loc:@Variable_13*
_output_shapes
: *

debug_nameVariable_13/*
dtype0*
shape:�*
shared_nameVariable_13
g
,Variable_13/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_13*
_output_shapes
: 
h
Variable_13/AssignAssignVariableOpVariable_13&Variable_13/Initializer/ReadVariableOp*
dtype0
h
Variable_13/Read/ReadVariableOpReadVariableOpVariable_13*
_output_shapes	
:�*
dtype0
�
'adam/batch_normalization_gamma_velocityVarHandleOp*
_output_shapes
: *8

debug_name*(adam/batch_normalization_gamma_velocity/*
dtype0*
shape:�*8
shared_name)'adam/batch_normalization_gamma_velocity
�
;adam/batch_normalization_gamma_velocity/Read/ReadVariableOpReadVariableOp'adam/batch_normalization_gamma_velocity*
_output_shapes	
:�*
dtype0
�
&Variable_14/Initializer/ReadVariableOpReadVariableOp'adam/batch_normalization_gamma_velocity*
_class
loc:@Variable_14*
_output_shapes	
:�*
dtype0
�
Variable_14VarHandleOp*
_class
loc:@Variable_14*
_output_shapes
: *

debug_nameVariable_14/*
dtype0*
shape:�*
shared_nameVariable_14
g
,Variable_14/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_14*
_output_shapes
: 
h
Variable_14/AssignAssignVariableOpVariable_14&Variable_14/Initializer/ReadVariableOp*
dtype0
h
Variable_14/Read/ReadVariableOpReadVariableOpVariable_14*
_output_shapes	
:�*
dtype0
�
'adam/batch_normalization_gamma_momentumVarHandleOp*
_output_shapes
: *8

debug_name*(adam/batch_normalization_gamma_momentum/*
dtype0*
shape:�*8
shared_name)'adam/batch_normalization_gamma_momentum
�
;adam/batch_normalization_gamma_momentum/Read/ReadVariableOpReadVariableOp'adam/batch_normalization_gamma_momentum*
_output_shapes	
:�*
dtype0
�
&Variable_15/Initializer/ReadVariableOpReadVariableOp'adam/batch_normalization_gamma_momentum*
_class
loc:@Variable_15*
_output_shapes	
:�*
dtype0
�
Variable_15VarHandleOp*
_class
loc:@Variable_15*
_output_shapes
: *

debug_nameVariable_15/*
dtype0*
shape:�*
shared_nameVariable_15
g
,Variable_15/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_15*
_output_shapes
: 
h
Variable_15/AssignAssignVariableOpVariable_15&Variable_15/Initializer/ReadVariableOp*
dtype0
h
Variable_15/Read/ReadVariableOpReadVariableOpVariable_15*
_output_shapes	
:�*
dtype0
�
adam/dense_bias_velocityVarHandleOp*
_output_shapes
: *)

debug_nameadam/dense_bias_velocity/*
dtype0*
shape:�*)
shared_nameadam/dense_bias_velocity
�
,adam/dense_bias_velocity/Read/ReadVariableOpReadVariableOpadam/dense_bias_velocity*
_output_shapes	
:�*
dtype0
�
&Variable_16/Initializer/ReadVariableOpReadVariableOpadam/dense_bias_velocity*
_class
loc:@Variable_16*
_output_shapes	
:�*
dtype0
�
Variable_16VarHandleOp*
_class
loc:@Variable_16*
_output_shapes
: *

debug_nameVariable_16/*
dtype0*
shape:�*
shared_nameVariable_16
g
,Variable_16/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_16*
_output_shapes
: 
h
Variable_16/AssignAssignVariableOpVariable_16&Variable_16/Initializer/ReadVariableOp*
dtype0
h
Variable_16/Read/ReadVariableOpReadVariableOpVariable_16*
_output_shapes	
:�*
dtype0
�
adam/dense_bias_momentumVarHandleOp*
_output_shapes
: *)

debug_nameadam/dense_bias_momentum/*
dtype0*
shape:�*)
shared_nameadam/dense_bias_momentum
�
,adam/dense_bias_momentum/Read/ReadVariableOpReadVariableOpadam/dense_bias_momentum*
_output_shapes	
:�*
dtype0
�
&Variable_17/Initializer/ReadVariableOpReadVariableOpadam/dense_bias_momentum*
_class
loc:@Variable_17*
_output_shapes	
:�*
dtype0
�
Variable_17VarHandleOp*
_class
loc:@Variable_17*
_output_shapes
: *

debug_nameVariable_17/*
dtype0*
shape:�*
shared_nameVariable_17
g
,Variable_17/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_17*
_output_shapes
: 
h
Variable_17/AssignAssignVariableOpVariable_17&Variable_17/Initializer/ReadVariableOp*
dtype0
h
Variable_17/Read/ReadVariableOpReadVariableOpVariable_17*
_output_shapes	
:�*
dtype0
�
adam/dense_kernel_velocityVarHandleOp*
_output_shapes
: *+

debug_nameadam/dense_kernel_velocity/*
dtype0*
shape:	�*+
shared_nameadam/dense_kernel_velocity
�
.adam/dense_kernel_velocity/Read/ReadVariableOpReadVariableOpadam/dense_kernel_velocity*
_output_shapes
:	�*
dtype0
�
&Variable_18/Initializer/ReadVariableOpReadVariableOpadam/dense_kernel_velocity*
_class
loc:@Variable_18*
_output_shapes
:	�*
dtype0
�
Variable_18VarHandleOp*
_class
loc:@Variable_18*
_output_shapes
: *

debug_nameVariable_18/*
dtype0*
shape:	�*
shared_nameVariable_18
g
,Variable_18/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_18*
_output_shapes
: 
h
Variable_18/AssignAssignVariableOpVariable_18&Variable_18/Initializer/ReadVariableOp*
dtype0
l
Variable_18/Read/ReadVariableOpReadVariableOpVariable_18*
_output_shapes
:	�*
dtype0
�
adam/dense_kernel_momentumVarHandleOp*
_output_shapes
: *+

debug_nameadam/dense_kernel_momentum/*
dtype0*
shape:	�*+
shared_nameadam/dense_kernel_momentum
�
.adam/dense_kernel_momentum/Read/ReadVariableOpReadVariableOpadam/dense_kernel_momentum*
_output_shapes
:	�*
dtype0
�
&Variable_19/Initializer/ReadVariableOpReadVariableOpadam/dense_kernel_momentum*
_class
loc:@Variable_19*
_output_shapes
:	�*
dtype0
�
Variable_19VarHandleOp*
_class
loc:@Variable_19*
_output_shapes
: *

debug_nameVariable_19/*
dtype0*
shape:	�*
shared_nameVariable_19
g
,Variable_19/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_19*
_output_shapes
: 
h
Variable_19/AssignAssignVariableOpVariable_19&Variable_19/Initializer/ReadVariableOp*
dtype0
l
Variable_19/Read/ReadVariableOpReadVariableOpVariable_19*
_output_shapes
:	�*
dtype0
�
dense_2/biasVarHandleOp*
_output_shapes
: *

debug_namedense_2/bias/*
dtype0*
shape:*
shared_namedense_2/bias
i
 dense_2/bias/Read/ReadVariableOpReadVariableOpdense_2/bias*
_output_shapes
:*
dtype0
�
&Variable_20/Initializer/ReadVariableOpReadVariableOpdense_2/bias*
_class
loc:@Variable_20*
_output_shapes
:*
dtype0
�
Variable_20VarHandleOp*
_class
loc:@Variable_20*
_output_shapes
: *

debug_nameVariable_20/*
dtype0*
shape:*
shared_nameVariable_20
g
,Variable_20/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_20*
_output_shapes
: 
h
Variable_20/AssignAssignVariableOpVariable_20&Variable_20/Initializer/ReadVariableOp*
dtype0
g
Variable_20/Read/ReadVariableOpReadVariableOpVariable_20*
_output_shapes
:*
dtype0
�
dense_2/kernelVarHandleOp*
_output_shapes
: *

debug_namedense_2/kernel/*
dtype0*
shape
:@*
shared_namedense_2/kernel
q
"dense_2/kernel/Read/ReadVariableOpReadVariableOpdense_2/kernel*
_output_shapes

:@*
dtype0
�
&Variable_21/Initializer/ReadVariableOpReadVariableOpdense_2/kernel*
_class
loc:@Variable_21*
_output_shapes

:@*
dtype0
�
Variable_21VarHandleOp*
_class
loc:@Variable_21*
_output_shapes
: *

debug_nameVariable_21/*
dtype0*
shape
:@*
shared_nameVariable_21
g
,Variable_21/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_21*
_output_shapes
: 
h
Variable_21/AssignAssignVariableOpVariable_21&Variable_21/Initializer/ReadVariableOp*
dtype0
k
Variable_21/Read/ReadVariableOpReadVariableOpVariable_21*
_output_shapes

:@*
dtype0
�
%batch_normalization_1/moving_varianceVarHandleOp*
_output_shapes
: *6

debug_name(&batch_normalization_1/moving_variance/*
dtype0*
shape:@*6
shared_name'%batch_normalization_1/moving_variance
�
9batch_normalization_1/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_1/moving_variance*
_output_shapes
:@*
dtype0
�
&Variable_22/Initializer/ReadVariableOpReadVariableOp%batch_normalization_1/moving_variance*
_class
loc:@Variable_22*
_output_shapes
:@*
dtype0
�
Variable_22VarHandleOp*
_class
loc:@Variable_22*
_output_shapes
: *

debug_nameVariable_22/*
dtype0*
shape:@*
shared_nameVariable_22
g
,Variable_22/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_22*
_output_shapes
: 
h
Variable_22/AssignAssignVariableOpVariable_22&Variable_22/Initializer/ReadVariableOp*
dtype0
g
Variable_22/Read/ReadVariableOpReadVariableOpVariable_22*
_output_shapes
:@*
dtype0
�
!batch_normalization_1/moving_meanVarHandleOp*
_output_shapes
: *2

debug_name$"batch_normalization_1/moving_mean/*
dtype0*
shape:@*2
shared_name#!batch_normalization_1/moving_mean
�
5batch_normalization_1/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_1/moving_mean*
_output_shapes
:@*
dtype0
�
&Variable_23/Initializer/ReadVariableOpReadVariableOp!batch_normalization_1/moving_mean*
_class
loc:@Variable_23*
_output_shapes
:@*
dtype0
�
Variable_23VarHandleOp*
_class
loc:@Variable_23*
_output_shapes
: *

debug_nameVariable_23/*
dtype0*
shape:@*
shared_nameVariable_23
g
,Variable_23/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_23*
_output_shapes
: 
h
Variable_23/AssignAssignVariableOpVariable_23&Variable_23/Initializer/ReadVariableOp*
dtype0
g
Variable_23/Read/ReadVariableOpReadVariableOpVariable_23*
_output_shapes
:@*
dtype0
�
batch_normalization_1/betaVarHandleOp*
_output_shapes
: *+

debug_namebatch_normalization_1/beta/*
dtype0*
shape:@*+
shared_namebatch_normalization_1/beta
�
.batch_normalization_1/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_1/beta*
_output_shapes
:@*
dtype0
�
&Variable_24/Initializer/ReadVariableOpReadVariableOpbatch_normalization_1/beta*
_class
loc:@Variable_24*
_output_shapes
:@*
dtype0
�
Variable_24VarHandleOp*
_class
loc:@Variable_24*
_output_shapes
: *

debug_nameVariable_24/*
dtype0*
shape:@*
shared_nameVariable_24
g
,Variable_24/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_24*
_output_shapes
: 
h
Variable_24/AssignAssignVariableOpVariable_24&Variable_24/Initializer/ReadVariableOp*
dtype0
g
Variable_24/Read/ReadVariableOpReadVariableOpVariable_24*
_output_shapes
:@*
dtype0
�
batch_normalization_1/gammaVarHandleOp*
_output_shapes
: *,

debug_namebatch_normalization_1/gamma/*
dtype0*
shape:@*,
shared_namebatch_normalization_1/gamma
�
/batch_normalization_1/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_1/gamma*
_output_shapes
:@*
dtype0
�
&Variable_25/Initializer/ReadVariableOpReadVariableOpbatch_normalization_1/gamma*
_class
loc:@Variable_25*
_output_shapes
:@*
dtype0
�
Variable_25VarHandleOp*
_class
loc:@Variable_25*
_output_shapes
: *

debug_nameVariable_25/*
dtype0*
shape:@*
shared_nameVariable_25
g
,Variable_25/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_25*
_output_shapes
: 
h
Variable_25/AssignAssignVariableOpVariable_25&Variable_25/Initializer/ReadVariableOp*
dtype0
g
Variable_25/Read/ReadVariableOpReadVariableOpVariable_25*
_output_shapes
:@*
dtype0
�
dense_1/biasVarHandleOp*
_output_shapes
: *

debug_namedense_1/bias/*
dtype0*
shape:@*
shared_namedense_1/bias
i
 dense_1/bias/Read/ReadVariableOpReadVariableOpdense_1/bias*
_output_shapes
:@*
dtype0
�
&Variable_26/Initializer/ReadVariableOpReadVariableOpdense_1/bias*
_class
loc:@Variable_26*
_output_shapes
:@*
dtype0
�
Variable_26VarHandleOp*
_class
loc:@Variable_26*
_output_shapes
: *

debug_nameVariable_26/*
dtype0*
shape:@*
shared_nameVariable_26
g
,Variable_26/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_26*
_output_shapes
: 
h
Variable_26/AssignAssignVariableOpVariable_26&Variable_26/Initializer/ReadVariableOp*
dtype0
g
Variable_26/Read/ReadVariableOpReadVariableOpVariable_26*
_output_shapes
:@*
dtype0
�
dense_1/kernelVarHandleOp*
_output_shapes
: *

debug_namedense_1/kernel/*
dtype0*
shape:	�@*
shared_namedense_1/kernel
r
"dense_1/kernel/Read/ReadVariableOpReadVariableOpdense_1/kernel*
_output_shapes
:	�@*
dtype0
�
&Variable_27/Initializer/ReadVariableOpReadVariableOpdense_1/kernel*
_class
loc:@Variable_27*
_output_shapes
:	�@*
dtype0
�
Variable_27VarHandleOp*
_class
loc:@Variable_27*
_output_shapes
: *

debug_nameVariable_27/*
dtype0*
shape:	�@*
shared_nameVariable_27
g
,Variable_27/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_27*
_output_shapes
: 
h
Variable_27/AssignAssignVariableOpVariable_27&Variable_27/Initializer/ReadVariableOp*
dtype0
l
Variable_27/Read/ReadVariableOpReadVariableOpVariable_27*
_output_shapes
:	�@*
dtype0
�
#batch_normalization/moving_varianceVarHandleOp*
_output_shapes
: *4

debug_name&$batch_normalization/moving_variance/*
dtype0*
shape:�*4
shared_name%#batch_normalization/moving_variance
�
7batch_normalization/moving_variance/Read/ReadVariableOpReadVariableOp#batch_normalization/moving_variance*
_output_shapes	
:�*
dtype0
�
&Variable_28/Initializer/ReadVariableOpReadVariableOp#batch_normalization/moving_variance*
_class
loc:@Variable_28*
_output_shapes	
:�*
dtype0
�
Variable_28VarHandleOp*
_class
loc:@Variable_28*
_output_shapes
: *

debug_nameVariable_28/*
dtype0*
shape:�*
shared_nameVariable_28
g
,Variable_28/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_28*
_output_shapes
: 
h
Variable_28/AssignAssignVariableOpVariable_28&Variable_28/Initializer/ReadVariableOp*
dtype0
h
Variable_28/Read/ReadVariableOpReadVariableOpVariable_28*
_output_shapes	
:�*
dtype0
�
batch_normalization/moving_meanVarHandleOp*
_output_shapes
: *0

debug_name" batch_normalization/moving_mean/*
dtype0*
shape:�*0
shared_name!batch_normalization/moving_mean
�
3batch_normalization/moving_mean/Read/ReadVariableOpReadVariableOpbatch_normalization/moving_mean*
_output_shapes	
:�*
dtype0
�
&Variable_29/Initializer/ReadVariableOpReadVariableOpbatch_normalization/moving_mean*
_class
loc:@Variable_29*
_output_shapes	
:�*
dtype0
�
Variable_29VarHandleOp*
_class
loc:@Variable_29*
_output_shapes
: *

debug_nameVariable_29/*
dtype0*
shape:�*
shared_nameVariable_29
g
,Variable_29/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_29*
_output_shapes
: 
h
Variable_29/AssignAssignVariableOpVariable_29&Variable_29/Initializer/ReadVariableOp*
dtype0
h
Variable_29/Read/ReadVariableOpReadVariableOpVariable_29*
_output_shapes	
:�*
dtype0
�
batch_normalization/betaVarHandleOp*
_output_shapes
: *)

debug_namebatch_normalization/beta/*
dtype0*
shape:�*)
shared_namebatch_normalization/beta
�
,batch_normalization/beta/Read/ReadVariableOpReadVariableOpbatch_normalization/beta*
_output_shapes	
:�*
dtype0
�
&Variable_30/Initializer/ReadVariableOpReadVariableOpbatch_normalization/beta*
_class
loc:@Variable_30*
_output_shapes	
:�*
dtype0
�
Variable_30VarHandleOp*
_class
loc:@Variable_30*
_output_shapes
: *

debug_nameVariable_30/*
dtype0*
shape:�*
shared_nameVariable_30
g
,Variable_30/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_30*
_output_shapes
: 
h
Variable_30/AssignAssignVariableOpVariable_30&Variable_30/Initializer/ReadVariableOp*
dtype0
h
Variable_30/Read/ReadVariableOpReadVariableOpVariable_30*
_output_shapes	
:�*
dtype0
�
batch_normalization/gammaVarHandleOp*
_output_shapes
: **

debug_namebatch_normalization/gamma/*
dtype0*
shape:�**
shared_namebatch_normalization/gamma
�
-batch_normalization/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization/gamma*
_output_shapes	
:�*
dtype0
�
&Variable_31/Initializer/ReadVariableOpReadVariableOpbatch_normalization/gamma*
_class
loc:@Variable_31*
_output_shapes	
:�*
dtype0
�
Variable_31VarHandleOp*
_class
loc:@Variable_31*
_output_shapes
: *

debug_nameVariable_31/*
dtype0*
shape:�*
shared_nameVariable_31
g
,Variable_31/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_31*
_output_shapes
: 
h
Variable_31/AssignAssignVariableOpVariable_31&Variable_31/Initializer/ReadVariableOp*
dtype0
h
Variable_31/Read/ReadVariableOpReadVariableOpVariable_31*
_output_shapes	
:�*
dtype0
�

dense/biasVarHandleOp*
_output_shapes
: *

debug_namedense/bias/*
dtype0*
shape:�*
shared_name
dense/bias
f
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
_output_shapes	
:�*
dtype0
�
&Variable_32/Initializer/ReadVariableOpReadVariableOp
dense/bias*
_class
loc:@Variable_32*
_output_shapes	
:�*
dtype0
�
Variable_32VarHandleOp*
_class
loc:@Variable_32*
_output_shapes
: *

debug_nameVariable_32/*
dtype0*
shape:�*
shared_nameVariable_32
g
,Variable_32/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_32*
_output_shapes
: 
h
Variable_32/AssignAssignVariableOpVariable_32&Variable_32/Initializer/ReadVariableOp*
dtype0
h
Variable_32/Read/ReadVariableOpReadVariableOpVariable_32*
_output_shapes	
:�*
dtype0
�
dense/kernelVarHandleOp*
_output_shapes
: *

debug_namedense/kernel/*
dtype0*
shape:	�*
shared_namedense/kernel
n
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel*
_output_shapes
:	�*
dtype0
�
&Variable_33/Initializer/ReadVariableOpReadVariableOpdense/kernel*
_class
loc:@Variable_33*
_output_shapes
:	�*
dtype0
�
Variable_33VarHandleOp*
_class
loc:@Variable_33*
_output_shapes
: *

debug_nameVariable_33/*
dtype0*
shape:	�*
shared_nameVariable_33
g
,Variable_33/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_33*
_output_shapes
: 
h
Variable_33/AssignAssignVariableOpVariable_33&Variable_33/Initializer/ReadVariableOp*
dtype0
l
Variable_33/Read/ReadVariableOpReadVariableOpVariable_33*
_output_shapes
:	�*
dtype0
�
adam/learning_rateVarHandleOp*
_output_shapes
: *#

debug_nameadam/learning_rate/*
dtype0*
shape: *#
shared_nameadam/learning_rate
q
&adam/learning_rate/Read/ReadVariableOpReadVariableOpadam/learning_rate*
_output_shapes
: *
dtype0
�
&Variable_34/Initializer/ReadVariableOpReadVariableOpadam/learning_rate*
_class
loc:@Variable_34*
_output_shapes
: *
dtype0
�
Variable_34VarHandleOp*
_class
loc:@Variable_34*
_output_shapes
: *

debug_nameVariable_34/*
dtype0*
shape: *
shared_nameVariable_34
g
,Variable_34/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_34*
_output_shapes
: 
h
Variable_34/AssignAssignVariableOpVariable_34&Variable_34/Initializer/ReadVariableOp*
dtype0
c
Variable_34/Read/ReadVariableOpReadVariableOpVariable_34*
_output_shapes
: *
dtype0
�
adam/iterationVarHandleOp*
_output_shapes
: *

debug_nameadam/iteration/*
dtype0	*
shape: *
shared_nameadam/iteration
i
"adam/iteration/Read/ReadVariableOpReadVariableOpadam/iteration*
_output_shapes
: *
dtype0	
�
&Variable_35/Initializer/ReadVariableOpReadVariableOpadam/iteration*
_class
loc:@Variable_35*
_output_shapes
: *
dtype0	
�
Variable_35VarHandleOp*
_class
loc:@Variable_35*
_output_shapes
: *

debug_nameVariable_35/*
dtype0	*
shape: *
shared_nameVariable_35
g
,Variable_35/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_35*
_output_shapes
: 
h
Variable_35/AssignAssignVariableOpVariable_35&Variable_35/Initializer/ReadVariableOp*
dtype0	
c
Variable_35/Read/ReadVariableOpReadVariableOpVariable_35*
_output_shapes
: *
dtype0	
s
serving_default_examplesPlaceholder*#
_output_shapes
:���������*
dtype0*
shape:���������
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_examplesConst_3Const_2Const_1Constdense/kernel
dense/biasbatch_normalization/moving_mean#batch_normalization/moving_variancebatch_normalization/gammabatch_normalization/betadense_1/kerneldense_1/bias!batch_normalization_1/moving_mean%batch_normalization_1/moving_variancebatch_normalization_1/gammabatch_normalization_1/betadense_2/kerneldense_2/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*0
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *,
f'R%
#__inference_signature_wrapper_47934

NoOpNoOp
�9
Const_4Const"/device:CPU:0*
_output_shapes
: *
dtype0*�9
value�9B�9 B�9
�
_tracked
_inbound_nodes
_outbound_nodes
_losses
_losses_override
_operations
_layers
_build_shapes_dict
	output_names

	optimizer
	tft_layer
_default_save_signature

signatures*
* 
* 
* 
* 
* 
Z
0
1
2
3
4
5
6
7
8
9
10
11*
Z
0
1
2
3
4
5
6
7
8
9
10
11*
* 
* 
�

_variables
_trainable_variables
 _trainable_variables_indices
_iterations
_learning_rate

_momentums
 _velocities*
�
!	variables
"trainable_variables
#regularization_losses
$	keras_api
%__call__
*&&call_and_return_all_conditional_losses
'_default_save_signature
$( _saved_model_loader_tracked_dict* 

)trace_0* 

*serving_default* 
]
+_inbound_nodes
,_outbound_nodes
-_losses
.	_loss_ids
/_losses_override* 
]
0_inbound_nodes
1_outbound_nodes
2_losses
3	_loss_ids
4_losses_override* 
]
5_inbound_nodes
6_outbound_nodes
7_losses
8	_loss_ids
9_losses_override* 
]
:_inbound_nodes
;_outbound_nodes
<_losses
=	_loss_ids
>_losses_override* 
u
?_inbound_nodes
@_outbound_nodes
A_losses
B	_loss_ids
C_losses_override
D_build_shapes_dict* 
�
E_kernel
Fbias
G_inbound_nodes
H_outbound_nodes
I_losses
J	_loss_ids
K_losses_override
L_build_shapes_dict*
�
	Mgamma
Nbeta
Omoving_mean
Pmoving_variance
Q_inbound_nodes
R_outbound_nodes
S_losses
T	_loss_ids
U_losses_override
V_reduction_axes
W_build_shapes_dict*
u
X_inbound_nodes
Y_outbound_nodes
Z_losses
[	_loss_ids
\_losses_override
]_build_shapes_dict* 
�
^_kernel
_bias
`_inbound_nodes
a_outbound_nodes
b_losses
c	_loss_ids
d_losses_override
e_build_shapes_dict*
�
	fgamma
gbeta
hmoving_mean
imoving_variance
j_inbound_nodes
k_outbound_nodes
l_losses
m	_loss_ids
n_losses_override
o_reduction_axes
p_build_shapes_dict*
u
q_inbound_nodes
r_outbound_nodes
s_losses
t	_loss_ids
u_losses_override
v_build_shapes_dict* 
�
w_kernel
xbias
y_inbound_nodes
z_outbound_nodes
{_losses
|	_loss_ids
}_losses_override
~_build_shapes_dict*
�
0
1
2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13
�14
�15
�16
�17
�18
�19
�20
�21*
J
E0
F1
M2
N3
^4
_5
f6
g7
w8
x9*
* 
UO
VARIABLE_VALUEVariable_350optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUE*
XR
VARIABLE_VALUEVariable_343optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
!	variables
"trainable_variables
#regularization_losses
%__call__
'_default_save_signature
*&&call_and_return_all_conditional_losses
&&"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 
B
�	capture_0
�	capture_1
�	capture_2
�	capture_3* 
y
�	_imported
�_wrapped_function
�_structured_inputs
�_structured_outputs
�_output_to_inputs_map* 
* 
B
�	capture_0
�	capture_1
�	capture_2
�	capture_3* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
UO
VARIABLE_VALUEVariable_330_operations/5/_kernel/.ATTRIBUTES/VARIABLE_VALUE*
RL
VARIABLE_VALUEVariable_32-_operations/5/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
* 
* 
SM
VARIABLE_VALUEVariable_31._operations/6/gamma/.ATTRIBUTES/VARIABLE_VALUE*
RL
VARIABLE_VALUEVariable_30-_operations/6/beta/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUEVariable_294_operations/6/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEVariable_288_operations/6/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
UO
VARIABLE_VALUEVariable_270_operations/8/_kernel/.ATTRIBUTES/VARIABLE_VALUE*
RL
VARIABLE_VALUEVariable_26-_operations/8/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
* 
* 
SM
VARIABLE_VALUEVariable_25._operations/9/gamma/.ATTRIBUTES/VARIABLE_VALUE*
RL
VARIABLE_VALUEVariable_24-_operations/9/beta/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUEVariable_234_operations/9/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEVariable_228_operations/9/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
VP
VARIABLE_VALUEVariable_211_operations/11/_kernel/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEVariable_20._operations/11/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
* 
* 
VP
VARIABLE_VALUEVariable_191optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUE*
VP
VARIABLE_VALUEVariable_181optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUE*
VP
VARIABLE_VALUEVariable_171optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUE*
VP
VARIABLE_VALUEVariable_161optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUE*
VP
VARIABLE_VALUEVariable_151optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUE*
VP
VARIABLE_VALUEVariable_141optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUE*
VP
VARIABLE_VALUEVariable_131optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUE*
VP
VARIABLE_VALUEVariable_121optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUE*
WQ
VARIABLE_VALUEVariable_112optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUE*
WQ
VARIABLE_VALUEVariable_102optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUE*
VP
VARIABLE_VALUE
Variable_92optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUE*
VP
VARIABLE_VALUE
Variable_82optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUE*
VP
VARIABLE_VALUE
Variable_72optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUE*
VP
VARIABLE_VALUE
Variable_62optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUE*
VP
VARIABLE_VALUE
Variable_52optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUE*
VP
VARIABLE_VALUE
Variable_42optimizer/_variables/17/.ATTRIBUTES/VARIABLE_VALUE*
VP
VARIABLE_VALUE
Variable_32optimizer/_variables/18/.ATTRIBUTES/VARIABLE_VALUE*
VP
VARIABLE_VALUE
Variable_22optimizer/_variables/19/.ATTRIBUTES/VARIABLE_VALUE*
VP
VARIABLE_VALUE
Variable_12optimizer/_variables/20/.ATTRIBUTES/VARIABLE_VALUE*
TN
VARIABLE_VALUEVariable2optimizer/_variables/21/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
* 
B
�	capture_0
�	capture_1
�	capture_2
�	capture_3* 
B
�	capture_0
�	capture_1
�	capture_2
�	capture_3* 
* 
* 
* 
* 
�
�created_variables
�	resources
�trackable_objects
�initializers
�assets
�
signatures
$�_self_saveable_object_factories
�transform_fn* 
B
�	capture_0
�	capture_1
�	capture_2
�	capture_3* 
* 
* 
* 
* 
* 
* 
* 
* 

�serving_default* 
* 
B
�	capture_0
�	capture_1
�	capture_2
�	capture_3* 
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenameVariable_35Variable_34Variable_33Variable_32Variable_31Variable_30Variable_29Variable_28Variable_27Variable_26Variable_25Variable_24Variable_23Variable_22Variable_21Variable_20Variable_19Variable_18Variable_17Variable_16Variable_15Variable_14Variable_13Variable_12Variable_11Variable_10
Variable_9
Variable_8
Variable_7
Variable_6
Variable_5
Variable_4
Variable_3
Variable_2
Variable_1VariableConst_4*1
Tin*
(2&*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *'
f"R 
__inference__traced_save_48481
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameVariable_35Variable_34Variable_33Variable_32Variable_31Variable_30Variable_29Variable_28Variable_27Variable_26Variable_25Variable_24Variable_23Variable_22Variable_21Variable_20Variable_19Variable_18Variable_17Variable_16Variable_15Variable_14Variable_13Variable_12Variable_11Variable_10
Variable_9
Variable_8
Variable_7
Variable_6
Variable_5
Variable_4
Variable_3
Variable_2
Variable_1Variable*0
Tin)
'2%*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� **
f%R#
!__inference__traced_restore_48598��
�$
�
__inference_pruned_47747

inputs	
inputs_1
inputs_2	
inputs_3	
inputs_4	
scale_to_z_score_sub_y
scale_to_z_score_sqrt_x
scale_to_z_score_1_sub_y
scale_to_z_score_1_sqrt_x
identity

identity_1

identity_2	

identity_3	

identity_4	`
scale_to_z_score/NotEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *    b
scale_to_z_score_1/NotEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *    Q
inputs_copyIdentityinputs*
T0	*'
_output_shapes
:���������t
scale_to_z_score/CastCastinputs_copy:output:0*

DstT0*

SrcT0	*'
_output_shapes
:����������
scale_to_z_score/subSubscale_to_z_score/Cast:y:0scale_to_z_score_sub_y*
T0*'
_output_shapes
:���������t
scale_to_z_score/zeros_like	ZerosLikescale_to_z_score/sub:z:0*
T0*'
_output_shapes
:���������W
scale_to_z_score/SqrtSqrtscale_to_z_score_sqrt_x*
T0*
_output_shapes
: �
scale_to_z_score/NotEqualNotEqualscale_to_z_score/Sqrt:y:0$scale_to_z_score/NotEqual/y:output:0*
T0*
_output_shapes
: n
scale_to_z_score/Cast_1Castscale_to_z_score/NotEqual:z:0*

DstT0*

SrcT0
*
_output_shapes
: �
scale_to_z_score/addAddV2scale_to_z_score/zeros_like:y:0scale_to_z_score/Cast_1:y:0*
T0*'
_output_shapes
:���������z
scale_to_z_score/Cast_2Castscale_to_z_score/add:z:0*

DstT0
*

SrcT0*'
_output_shapes
:����������
scale_to_z_score/truedivRealDivscale_to_z_score/sub:z:0scale_to_z_score/Sqrt:y:0*
T0*'
_output_shapes
:����������
scale_to_z_score/SelectV2SelectV2scale_to_z_score/Cast_2:y:0scale_to_z_score/truediv:z:0scale_to_z_score/sub:z:0*
T0*'
_output_shapes
:���������j
IdentityIdentity"scale_to_z_score/SelectV2:output:0*
T0*'
_output_shapes
:���������U
inputs_1_copyIdentityinputs_1*
T0*'
_output_shapes
:����������
scale_to_z_score_1/subSubinputs_1_copy:output:0scale_to_z_score_1_sub_y*
T0*'
_output_shapes
:���������x
scale_to_z_score_1/zeros_like	ZerosLikescale_to_z_score_1/sub:z:0*
T0*'
_output_shapes
:���������[
scale_to_z_score_1/SqrtSqrtscale_to_z_score_1_sqrt_x*
T0*
_output_shapes
: �
scale_to_z_score_1/NotEqualNotEqualscale_to_z_score_1/Sqrt:y:0&scale_to_z_score_1/NotEqual/y:output:0*
T0*
_output_shapes
: p
scale_to_z_score_1/CastCastscale_to_z_score_1/NotEqual:z:0*

DstT0*

SrcT0
*
_output_shapes
: �
scale_to_z_score_1/addAddV2!scale_to_z_score_1/zeros_like:y:0scale_to_z_score_1/Cast:y:0*
T0*'
_output_shapes
:���������~
scale_to_z_score_1/Cast_1Castscale_to_z_score_1/add:z:0*

DstT0
*

SrcT0*'
_output_shapes
:����������
scale_to_z_score_1/truedivRealDivscale_to_z_score_1/sub:z:0scale_to_z_score_1/Sqrt:y:0*
T0*'
_output_shapes
:����������
scale_to_z_score_1/SelectV2SelectV2scale_to_z_score_1/Cast_1:y:0scale_to_z_score_1/truediv:z:0scale_to_z_score_1/sub:z:0*
T0*'
_output_shapes
:���������n

Identity_1Identity$scale_to_z_score_1/SelectV2:output:0*
T0*'
_output_shapes
:���������U
inputs_2_copyIdentityinputs_2*
T0	*'
_output_shapes
:���������`

Identity_2Identityinputs_2_copy:output:0*
T0	*'
_output_shapes
:���������U
inputs_3_copyIdentityinputs_3*
T0	*'
_output_shapes
:���������`

Identity_3Identityinputs_3_copy:output:0*
T0	*'
_output_shapes
:���������U
inputs_4_copyIdentityinputs_4*
T0	*'
_output_shapes
:���������`

Identity_4Identityinputs_4_copy:output:0*
T0	*'
_output_shapes
:���������"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*(
_construction_contextkEagerRuntime*z
_input_shapesi
g:���������:���������:���������:���������:���������: : : : :- )
'
_output_shapes
:���������:-)
'
_output_shapes
:���������:-)
'
_output_shapes
:���������:-)
'
_output_shapes
:���������:-)
'
_output_shapes
:���������:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
�
�
8__inference_transform_features_layer_layer_call_fn_48095
age	
bmi

gender	
physicalactivitylevel	
unknown
	unknown_0
	unknown_1
	unknown_2
identity

identity_1

identity_2	

identity_3	�
PartitionedCallPartitionedCallagebmigenderphysicalactivitylevelunknown	unknown_0	unknown_1	unknown_2*
Tin

2			*
Tout
2		*
_collective_manager_ids
 *`
_output_shapesN
L:���������:���������:���������:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *\
fWRU
S__inference_transform_features_layer_layer_call_and_return_conditional_losses_48073`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:���������b

Identity_1IdentityPartitionedCall:output:1*
T0*'
_output_shapes
:���������b

Identity_2IdentityPartitionedCall:output:2*
T0	*'
_output_shapes
:���������b

Identity_3IdentityPartitionedCall:output:3*
T0	*'
_output_shapes
:���������"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*(
_construction_contextkEagerRuntime*g
_input_shapesV
T:���������:���������:���������:���������: : : : :L H
'
_output_shapes
:���������

_user_specified_nameAge:LH
'
_output_shapes
:���������

_user_specified_nameBMI:OK
'
_output_shapes
:���������
 
_user_specified_nameGender:^Z
'
_output_shapes
:���������
/
_user_specified_namePhysicalActivityLevel:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
�d
�
!__inference_serving_default_47996

inputs
inputs_1
inputs_2
inputs_3D
1functional_1_dense_1_cast_readvariableop_resource:	�?
0functional_1_dense_1_add_readvariableop_resource:	�N
?functional_1_batch_normalization_1_cast_readvariableop_resource:	�P
Afunctional_1_batch_normalization_1_cast_1_readvariableop_resource:	�P
Afunctional_1_batch_normalization_1_cast_2_readvariableop_resource:	�P
Afunctional_1_batch_normalization_1_cast_3_readvariableop_resource:	�F
3functional_1_dense_1_2_cast_readvariableop_resource:	�@@
2functional_1_dense_1_2_add_readvariableop_resource:@O
Afunctional_1_batch_normalization_1_2_cast_readvariableop_resource:@Q
Cfunctional_1_batch_normalization_1_2_cast_1_readvariableop_resource:@Q
Cfunctional_1_batch_normalization_1_2_cast_2_readvariableop_resource:@Q
Cfunctional_1_batch_normalization_1_2_cast_3_readvariableop_resource:@E
3functional_1_dense_2_1_cast_readvariableop_resource:@@
2functional_1_dense_2_1_add_readvariableop_resource:
identity��6functional_1/batch_normalization_1/Cast/ReadVariableOp�8functional_1/batch_normalization_1/Cast_1/ReadVariableOp�8functional_1/batch_normalization_1/Cast_2/ReadVariableOp�8functional_1/batch_normalization_1/Cast_3/ReadVariableOp�8functional_1/batch_normalization_1_2/Cast/ReadVariableOp�:functional_1/batch_normalization_1_2/Cast_1/ReadVariableOp�:functional_1/batch_normalization_1_2/Cast_2/ReadVariableOp�:functional_1/batch_normalization_1_2/Cast_3/ReadVariableOp�'functional_1/dense_1/Add/ReadVariableOp�(functional_1/dense_1/Cast/ReadVariableOp�)functional_1/dense_1_2/Add/ReadVariableOp�*functional_1/dense_1_2/Cast/ReadVariableOp�)functional_1/dense_2_1/Add/ReadVariableOp�*functional_1/dense_2_1/Cast/ReadVariableOpq
&functional_1/concatenate_1/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
����������
!functional_1/concatenate_1/concatConcatV2inputsinputs_1inputs_2inputs_3/functional_1/concatenate_1/concat/axis:output:0*
N*
T0*'
_output_shapes
:����������
(functional_1/dense_1/Cast/ReadVariableOpReadVariableOp1functional_1_dense_1_cast_readvariableop_resource*
_output_shapes
:	�*
dtype0�
functional_1/dense_1/MatMulMatMul*functional_1/concatenate_1/concat:output:00functional_1/dense_1/Cast/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
'functional_1/dense_1/Add/ReadVariableOpReadVariableOp0functional_1_dense_1_add_readvariableop_resource*
_output_shapes	
:�*
dtype0�
functional_1/dense_1/AddAddV2%functional_1/dense_1/MatMul:product:0/functional_1/dense_1/Add/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������r
functional_1/dense_1/ReluRelufunctional_1/dense_1/Add:z:0*
T0*(
_output_shapes
:�����������
6functional_1/batch_normalization_1/Cast/ReadVariableOpReadVariableOp?functional_1_batch_normalization_1_cast_readvariableop_resource*
_output_shapes	
:�*
dtype0�
8functional_1/batch_normalization_1/Cast_1/ReadVariableOpReadVariableOpAfunctional_1_batch_normalization_1_cast_1_readvariableop_resource*
_output_shapes	
:�*
dtype0�
8functional_1/batch_normalization_1/Cast_2/ReadVariableOpReadVariableOpAfunctional_1_batch_normalization_1_cast_2_readvariableop_resource*
_output_shapes	
:�*
dtype0�
8functional_1/batch_normalization_1/Cast_3/ReadVariableOpReadVariableOpAfunctional_1_batch_normalization_1_cast_3_readvariableop_resource*
_output_shapes	
:�*
dtype0w
2functional_1/batch_normalization_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
0functional_1/batch_normalization_1/batchnorm/addAddV2@functional_1/batch_normalization_1/Cast_1/ReadVariableOp:value:0;functional_1/batch_normalization_1/batchnorm/add/y:output:0*
T0*
_output_shapes	
:��
2functional_1/batch_normalization_1/batchnorm/RsqrtRsqrt4functional_1/batch_normalization_1/batchnorm/add:z:0*
T0*
_output_shapes	
:��
0functional_1/batch_normalization_1/batchnorm/mulMul6functional_1/batch_normalization_1/batchnorm/Rsqrt:y:0@functional_1/batch_normalization_1/Cast_2/ReadVariableOp:value:0*
T0*
_output_shapes	
:��
2functional_1/batch_normalization_1/batchnorm/mul_1Mul'functional_1/dense_1/Relu:activations:04functional_1/batch_normalization_1/batchnorm/mul:z:0*
T0*(
_output_shapes
:�����������
2functional_1/batch_normalization_1/batchnorm/mul_2Mul>functional_1/batch_normalization_1/Cast/ReadVariableOp:value:04functional_1/batch_normalization_1/batchnorm/mul:z:0*
T0*
_output_shapes	
:��
0functional_1/batch_normalization_1/batchnorm/subSub@functional_1/batch_normalization_1/Cast_3/ReadVariableOp:value:06functional_1/batch_normalization_1/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:��
2functional_1/batch_normalization_1/batchnorm/add_1AddV26functional_1/batch_normalization_1/batchnorm/mul_1:z:04functional_1/batch_normalization_1/batchnorm/sub:z:0*
T0*(
_output_shapes
:�����������
*functional_1/dense_1_2/Cast/ReadVariableOpReadVariableOp3functional_1_dense_1_2_cast_readvariableop_resource*
_output_shapes
:	�@*
dtype0�
functional_1/dense_1_2/MatMulMatMul6functional_1/batch_normalization_1/batchnorm/add_1:z:02functional_1/dense_1_2/Cast/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
)functional_1/dense_1_2/Add/ReadVariableOpReadVariableOp2functional_1_dense_1_2_add_readvariableop_resource*
_output_shapes
:@*
dtype0�
functional_1/dense_1_2/AddAddV2'functional_1/dense_1_2/MatMul:product:01functional_1/dense_1_2/Add/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@u
functional_1/dense_1_2/ReluRelufunctional_1/dense_1_2/Add:z:0*
T0*'
_output_shapes
:���������@�
8functional_1/batch_normalization_1_2/Cast/ReadVariableOpReadVariableOpAfunctional_1_batch_normalization_1_2_cast_readvariableop_resource*
_output_shapes
:@*
dtype0�
:functional_1/batch_normalization_1_2/Cast_1/ReadVariableOpReadVariableOpCfunctional_1_batch_normalization_1_2_cast_1_readvariableop_resource*
_output_shapes
:@*
dtype0�
:functional_1/batch_normalization_1_2/Cast_2/ReadVariableOpReadVariableOpCfunctional_1_batch_normalization_1_2_cast_2_readvariableop_resource*
_output_shapes
:@*
dtype0�
:functional_1/batch_normalization_1_2/Cast_3/ReadVariableOpReadVariableOpCfunctional_1_batch_normalization_1_2_cast_3_readvariableop_resource*
_output_shapes
:@*
dtype0y
4functional_1/batch_normalization_1_2/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
2functional_1/batch_normalization_1_2/batchnorm/addAddV2Bfunctional_1/batch_normalization_1_2/Cast_1/ReadVariableOp:value:0=functional_1/batch_normalization_1_2/batchnorm/add/y:output:0*
T0*
_output_shapes
:@�
4functional_1/batch_normalization_1_2/batchnorm/RsqrtRsqrt6functional_1/batch_normalization_1_2/batchnorm/add:z:0*
T0*
_output_shapes
:@�
2functional_1/batch_normalization_1_2/batchnorm/mulMul8functional_1/batch_normalization_1_2/batchnorm/Rsqrt:y:0Bfunctional_1/batch_normalization_1_2/Cast_2/ReadVariableOp:value:0*
T0*
_output_shapes
:@�
4functional_1/batch_normalization_1_2/batchnorm/mul_1Mul)functional_1/dense_1_2/Relu:activations:06functional_1/batch_normalization_1_2/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������@�
4functional_1/batch_normalization_1_2/batchnorm/mul_2Mul@functional_1/batch_normalization_1_2/Cast/ReadVariableOp:value:06functional_1/batch_normalization_1_2/batchnorm/mul:z:0*
T0*
_output_shapes
:@�
2functional_1/batch_normalization_1_2/batchnorm/subSubBfunctional_1/batch_normalization_1_2/Cast_3/ReadVariableOp:value:08functional_1/batch_normalization_1_2/batchnorm/mul_2:z:0*
T0*
_output_shapes
:@�
4functional_1/batch_normalization_1_2/batchnorm/add_1AddV28functional_1/batch_normalization_1_2/batchnorm/mul_1:z:06functional_1/batch_normalization_1_2/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������@�
*functional_1/dense_2_1/Cast/ReadVariableOpReadVariableOp3functional_1_dense_2_1_cast_readvariableop_resource*
_output_shapes

:@*
dtype0�
functional_1/dense_2_1/MatMulMatMul8functional_1/batch_normalization_1_2/batchnorm/add_1:z:02functional_1/dense_2_1/Cast/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
)functional_1/dense_2_1/Add/ReadVariableOpReadVariableOp2functional_1_dense_2_1_add_readvariableop_resource*
_output_shapes
:*
dtype0�
functional_1/dense_2_1/AddAddV2'functional_1/dense_2_1/MatMul:product:01functional_1/dense_2_1/Add/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������{
functional_1/dense_2_1/SoftmaxSoftmaxfunctional_1/dense_2_1/Add:z:0*
T0*'
_output_shapes
:���������w
IdentityIdentity(functional_1/dense_2_1/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp7^functional_1/batch_normalization_1/Cast/ReadVariableOp9^functional_1/batch_normalization_1/Cast_1/ReadVariableOp9^functional_1/batch_normalization_1/Cast_2/ReadVariableOp9^functional_1/batch_normalization_1/Cast_3/ReadVariableOp9^functional_1/batch_normalization_1_2/Cast/ReadVariableOp;^functional_1/batch_normalization_1_2/Cast_1/ReadVariableOp;^functional_1/batch_normalization_1_2/Cast_2/ReadVariableOp;^functional_1/batch_normalization_1_2/Cast_3/ReadVariableOp(^functional_1/dense_1/Add/ReadVariableOp)^functional_1/dense_1/Cast/ReadVariableOp*^functional_1/dense_1_2/Add/ReadVariableOp+^functional_1/dense_1_2/Cast/ReadVariableOp*^functional_1/dense_2_1/Add/ReadVariableOp+^functional_1/dense_2_1/Cast/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*{
_input_shapesj
h:���������:���������:���������:���������: : : : : : : : : : : : : : 2p
6functional_1/batch_normalization_1/Cast/ReadVariableOp6functional_1/batch_normalization_1/Cast/ReadVariableOp2t
8functional_1/batch_normalization_1/Cast_1/ReadVariableOp8functional_1/batch_normalization_1/Cast_1/ReadVariableOp2t
8functional_1/batch_normalization_1/Cast_2/ReadVariableOp8functional_1/batch_normalization_1/Cast_2/ReadVariableOp2t
8functional_1/batch_normalization_1/Cast_3/ReadVariableOp8functional_1/batch_normalization_1/Cast_3/ReadVariableOp2t
8functional_1/batch_normalization_1_2/Cast/ReadVariableOp8functional_1/batch_normalization_1_2/Cast/ReadVariableOp2x
:functional_1/batch_normalization_1_2/Cast_1/ReadVariableOp:functional_1/batch_normalization_1_2/Cast_1/ReadVariableOp2x
:functional_1/batch_normalization_1_2/Cast_2/ReadVariableOp:functional_1/batch_normalization_1_2/Cast_2/ReadVariableOp2x
:functional_1/batch_normalization_1_2/Cast_3/ReadVariableOp:functional_1/batch_normalization_1_2/Cast_3/ReadVariableOp2R
'functional_1/dense_1/Add/ReadVariableOp'functional_1/dense_1/Add/ReadVariableOp2T
(functional_1/dense_1/Cast/ReadVariableOp(functional_1/dense_1/Cast/ReadVariableOp2V
)functional_1/dense_1_2/Add/ReadVariableOp)functional_1/dense_1_2/Add/ReadVariableOp2X
*functional_1/dense_1_2/Cast/ReadVariableOp*functional_1/dense_1_2/Cast/ReadVariableOp2V
)functional_1/dense_2_1/Add/ReadVariableOp)functional_1/dense_2_1/Add/ReadVariableOp2X
*functional_1/dense_2_1/Cast/ReadVariableOp*functional_1/dense_2_1/Cast/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs:OK
'
_output_shapes
:���������
 
_user_specified_nameinputs:OK
'
_output_shapes
:���������
 
_user_specified_nameinputs:OK
'
_output_shapes
:���������
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:(	$
"
_user_specified_name
resource:(
$
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�
�
S__inference_transform_features_layer_layer_call_and_return_conditional_losses_48073
age	
bmi

gender	
physicalactivitylevel	
unknown
	unknown_0
	unknown_1
	unknown_2
identity

identity_1

identity_2	

identity_3	F
ShapeShapeage*
T0	*
_output_shapes
::��]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskH
Shape_1Shapeage*
T0	*
_output_shapes
::��_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :u
zeros/packedPackstrided_slice_1:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:M
zeros/ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0	*'
_output_shapes
:����������
PlaceholderWithDefaultPlaceholderWithDefaultzeros:output:0*'
_output_shapes
:���������*
dtype0	*
shape:����������
PartitionedCallPartitionedCallagebmigenderPlaceholderWithDefault:output:0physicalactivitylevelunknown	unknown_0	unknown_1	unknown_2*
Tin
2					*
Tout	
2			*
_collective_manager_ids
 *s
_output_shapesa
_:���������:���������:���������:���������:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *!
fR
__inference_pruned_47747`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:���������b

Identity_1IdentityPartitionedCall:output:1*
T0*'
_output_shapes
:���������b

Identity_2IdentityPartitionedCall:output:2*
T0	*'
_output_shapes
:���������b

Identity_3IdentityPartitionedCall:output:4*
T0	*'
_output_shapes
:���������"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*(
_construction_contextkEagerRuntime*g
_input_shapesV
T:���������:���������:���������:���������: : : : :L H
'
_output_shapes
:���������

_user_specified_nameAge:LH
'
_output_shapes
:���������

_user_specified_nameBMI:OK
'
_output_shapes
:���������
 
_user_specified_nameGender:^Z
'
_output_shapes
:���������
/
_user_specified_namePhysicalActivityLevel:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
�
�
#__inference_signature_wrapper_47768

inputs	
inputs_1
inputs_2	
inputs_3	
inputs_4	
unknown
	unknown_0
	unknown_1
	unknown_2
identity

identity_1

identity_2	

identity_3	

identity_4	�
PartitionedCallPartitionedCallinputsinputs_1inputs_2inputs_3inputs_4unknown	unknown_0	unknown_1	unknown_2*
Tin
2					*
Tout	
2			*
_collective_manager_ids
 *s
_output_shapesa
_:���������:���������:���������:���������:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *!
fR
__inference_pruned_47747`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:���������b

Identity_1IdentityPartitionedCall:output:1*
T0*'
_output_shapes
:���������b

Identity_2IdentityPartitionedCall:output:2*
T0	*'
_output_shapes
:���������b

Identity_3IdentityPartitionedCall:output:3*
T0	*'
_output_shapes
:���������b

Identity_4IdentityPartitionedCall:output:4*
T0	*'
_output_shapes
:���������"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*(
_construction_contextkEagerRuntime*z
_input_shapesi
g:���������:���������:���������:���������:���������: : : : :O K
'
_output_shapes
:���������
 
_user_specified_nameinputs:QM
'
_output_shapes
:���������
"
_user_specified_name
inputs_1:QM
'
_output_shapes
:���������
"
_user_specified_name
inputs_2:QM
'
_output_shapes
:���������
"
_user_specified_name
inputs_3:QM
'
_output_shapes
:���������
"
_user_specified_name
inputs_4:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
��
�
__inference__traced_save_48481
file_prefix,
"read_disablecopyonread_variable_35:	 .
$read_1_disablecopyonread_variable_34: 7
$read_2_disablecopyonread_variable_33:	�3
$read_3_disablecopyonread_variable_32:	�3
$read_4_disablecopyonread_variable_31:	�3
$read_5_disablecopyonread_variable_30:	�3
$read_6_disablecopyonread_variable_29:	�3
$read_7_disablecopyonread_variable_28:	�7
$read_8_disablecopyonread_variable_27:	�@2
$read_9_disablecopyonread_variable_26:@3
%read_10_disablecopyonread_variable_25:@3
%read_11_disablecopyonread_variable_24:@3
%read_12_disablecopyonread_variable_23:@3
%read_13_disablecopyonread_variable_22:@7
%read_14_disablecopyonread_variable_21:@3
%read_15_disablecopyonread_variable_20:8
%read_16_disablecopyonread_variable_19:	�8
%read_17_disablecopyonread_variable_18:	�4
%read_18_disablecopyonread_variable_17:	�4
%read_19_disablecopyonread_variable_16:	�4
%read_20_disablecopyonread_variable_15:	�4
%read_21_disablecopyonread_variable_14:	�4
%read_22_disablecopyonread_variable_13:	�4
%read_23_disablecopyonread_variable_12:	�8
%read_24_disablecopyonread_variable_11:	�@8
%read_25_disablecopyonread_variable_10:	�@2
$read_26_disablecopyonread_variable_9:@2
$read_27_disablecopyonread_variable_8:@2
$read_28_disablecopyonread_variable_7:@2
$read_29_disablecopyonread_variable_6:@2
$read_30_disablecopyonread_variable_5:@2
$read_31_disablecopyonread_variable_4:@6
$read_32_disablecopyonread_variable_3:@6
$read_33_disablecopyonread_variable_2:@2
$read_34_disablecopyonread_variable_1:0
"read_35_disablecopyonread_variable:
savev2_const_4
identity_73��MergeV2Checkpoints�Read/DisableCopyOnRead�Read/ReadVariableOp�Read_1/DisableCopyOnRead�Read_1/ReadVariableOp�Read_10/DisableCopyOnRead�Read_10/ReadVariableOp�Read_11/DisableCopyOnRead�Read_11/ReadVariableOp�Read_12/DisableCopyOnRead�Read_12/ReadVariableOp�Read_13/DisableCopyOnRead�Read_13/ReadVariableOp�Read_14/DisableCopyOnRead�Read_14/ReadVariableOp�Read_15/DisableCopyOnRead�Read_15/ReadVariableOp�Read_16/DisableCopyOnRead�Read_16/ReadVariableOp�Read_17/DisableCopyOnRead�Read_17/ReadVariableOp�Read_18/DisableCopyOnRead�Read_18/ReadVariableOp�Read_19/DisableCopyOnRead�Read_19/ReadVariableOp�Read_2/DisableCopyOnRead�Read_2/ReadVariableOp�Read_20/DisableCopyOnRead�Read_20/ReadVariableOp�Read_21/DisableCopyOnRead�Read_21/ReadVariableOp�Read_22/DisableCopyOnRead�Read_22/ReadVariableOp�Read_23/DisableCopyOnRead�Read_23/ReadVariableOp�Read_24/DisableCopyOnRead�Read_24/ReadVariableOp�Read_25/DisableCopyOnRead�Read_25/ReadVariableOp�Read_26/DisableCopyOnRead�Read_26/ReadVariableOp�Read_27/DisableCopyOnRead�Read_27/ReadVariableOp�Read_28/DisableCopyOnRead�Read_28/ReadVariableOp�Read_29/DisableCopyOnRead�Read_29/ReadVariableOp�Read_3/DisableCopyOnRead�Read_3/ReadVariableOp�Read_30/DisableCopyOnRead�Read_30/ReadVariableOp�Read_31/DisableCopyOnRead�Read_31/ReadVariableOp�Read_32/DisableCopyOnRead�Read_32/ReadVariableOp�Read_33/DisableCopyOnRead�Read_33/ReadVariableOp�Read_34/DisableCopyOnRead�Read_34/ReadVariableOp�Read_35/DisableCopyOnRead�Read_35/ReadVariableOp�Read_4/DisableCopyOnRead�Read_4/ReadVariableOp�Read_5/DisableCopyOnRead�Read_5/ReadVariableOp�Read_6/DisableCopyOnRead�Read_6/ReadVariableOp�Read_7/DisableCopyOnRead�Read_7/ReadVariableOp�Read_8/DisableCopyOnRead�Read_8/ReadVariableOp�Read_9/DisableCopyOnRead�Read_9/ReadVariableOpw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part�
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: e
Read/DisableCopyOnReadDisableCopyOnRead"read_disablecopyonread_variable_35*
_output_shapes
 �
Read/ReadVariableOpReadVariableOp"read_disablecopyonread_variable_35^Read/DisableCopyOnRead*
_output_shapes
: *
dtype0	R
IdentityIdentityRead/ReadVariableOp:value:0*
T0	*
_output_shapes
: Y

Identity_1IdentityIdentity:output:0"/device:CPU:0*
T0	*
_output_shapes
: i
Read_1/DisableCopyOnReadDisableCopyOnRead$read_1_disablecopyonread_variable_34*
_output_shapes
 �
Read_1/ReadVariableOpReadVariableOp$read_1_disablecopyonread_variable_34^Read_1/DisableCopyOnRead*
_output_shapes
: *
dtype0V

Identity_2IdentityRead_1/ReadVariableOp:value:0*
T0*
_output_shapes
: [

Identity_3IdentityIdentity_2:output:0"/device:CPU:0*
T0*
_output_shapes
: i
Read_2/DisableCopyOnReadDisableCopyOnRead$read_2_disablecopyonread_variable_33*
_output_shapes
 �
Read_2/ReadVariableOpReadVariableOp$read_2_disablecopyonread_variable_33^Read_2/DisableCopyOnRead*
_output_shapes
:	�*
dtype0_

Identity_4IdentityRead_2/ReadVariableOp:value:0*
T0*
_output_shapes
:	�d

Identity_5IdentityIdentity_4:output:0"/device:CPU:0*
T0*
_output_shapes
:	�i
Read_3/DisableCopyOnReadDisableCopyOnRead$read_3_disablecopyonread_variable_32*
_output_shapes
 �
Read_3/ReadVariableOpReadVariableOp$read_3_disablecopyonread_variable_32^Read_3/DisableCopyOnRead*
_output_shapes	
:�*
dtype0[

Identity_6IdentityRead_3/ReadVariableOp:value:0*
T0*
_output_shapes	
:�`

Identity_7IdentityIdentity_6:output:0"/device:CPU:0*
T0*
_output_shapes	
:�i
Read_4/DisableCopyOnReadDisableCopyOnRead$read_4_disablecopyonread_variable_31*
_output_shapes
 �
Read_4/ReadVariableOpReadVariableOp$read_4_disablecopyonread_variable_31^Read_4/DisableCopyOnRead*
_output_shapes	
:�*
dtype0[

Identity_8IdentityRead_4/ReadVariableOp:value:0*
T0*
_output_shapes	
:�`

Identity_9IdentityIdentity_8:output:0"/device:CPU:0*
T0*
_output_shapes	
:�i
Read_5/DisableCopyOnReadDisableCopyOnRead$read_5_disablecopyonread_variable_30*
_output_shapes
 �
Read_5/ReadVariableOpReadVariableOp$read_5_disablecopyonread_variable_30^Read_5/DisableCopyOnRead*
_output_shapes	
:�*
dtype0\
Identity_10IdentityRead_5/ReadVariableOp:value:0*
T0*
_output_shapes	
:�b
Identity_11IdentityIdentity_10:output:0"/device:CPU:0*
T0*
_output_shapes	
:�i
Read_6/DisableCopyOnReadDisableCopyOnRead$read_6_disablecopyonread_variable_29*
_output_shapes
 �
Read_6/ReadVariableOpReadVariableOp$read_6_disablecopyonread_variable_29^Read_6/DisableCopyOnRead*
_output_shapes	
:�*
dtype0\
Identity_12IdentityRead_6/ReadVariableOp:value:0*
T0*
_output_shapes	
:�b
Identity_13IdentityIdentity_12:output:0"/device:CPU:0*
T0*
_output_shapes	
:�i
Read_7/DisableCopyOnReadDisableCopyOnRead$read_7_disablecopyonread_variable_28*
_output_shapes
 �
Read_7/ReadVariableOpReadVariableOp$read_7_disablecopyonread_variable_28^Read_7/DisableCopyOnRead*
_output_shapes	
:�*
dtype0\
Identity_14IdentityRead_7/ReadVariableOp:value:0*
T0*
_output_shapes	
:�b
Identity_15IdentityIdentity_14:output:0"/device:CPU:0*
T0*
_output_shapes	
:�i
Read_8/DisableCopyOnReadDisableCopyOnRead$read_8_disablecopyonread_variable_27*
_output_shapes
 �
Read_8/ReadVariableOpReadVariableOp$read_8_disablecopyonread_variable_27^Read_8/DisableCopyOnRead*
_output_shapes
:	�@*
dtype0`
Identity_16IdentityRead_8/ReadVariableOp:value:0*
T0*
_output_shapes
:	�@f
Identity_17IdentityIdentity_16:output:0"/device:CPU:0*
T0*
_output_shapes
:	�@i
Read_9/DisableCopyOnReadDisableCopyOnRead$read_9_disablecopyonread_variable_26*
_output_shapes
 �
Read_9/ReadVariableOpReadVariableOp$read_9_disablecopyonread_variable_26^Read_9/DisableCopyOnRead*
_output_shapes
:@*
dtype0[
Identity_18IdentityRead_9/ReadVariableOp:value:0*
T0*
_output_shapes
:@a
Identity_19IdentityIdentity_18:output:0"/device:CPU:0*
T0*
_output_shapes
:@k
Read_10/DisableCopyOnReadDisableCopyOnRead%read_10_disablecopyonread_variable_25*
_output_shapes
 �
Read_10/ReadVariableOpReadVariableOp%read_10_disablecopyonread_variable_25^Read_10/DisableCopyOnRead*
_output_shapes
:@*
dtype0\
Identity_20IdentityRead_10/ReadVariableOp:value:0*
T0*
_output_shapes
:@a
Identity_21IdentityIdentity_20:output:0"/device:CPU:0*
T0*
_output_shapes
:@k
Read_11/DisableCopyOnReadDisableCopyOnRead%read_11_disablecopyonread_variable_24*
_output_shapes
 �
Read_11/ReadVariableOpReadVariableOp%read_11_disablecopyonread_variable_24^Read_11/DisableCopyOnRead*
_output_shapes
:@*
dtype0\
Identity_22IdentityRead_11/ReadVariableOp:value:0*
T0*
_output_shapes
:@a
Identity_23IdentityIdentity_22:output:0"/device:CPU:0*
T0*
_output_shapes
:@k
Read_12/DisableCopyOnReadDisableCopyOnRead%read_12_disablecopyonread_variable_23*
_output_shapes
 �
Read_12/ReadVariableOpReadVariableOp%read_12_disablecopyonread_variable_23^Read_12/DisableCopyOnRead*
_output_shapes
:@*
dtype0\
Identity_24IdentityRead_12/ReadVariableOp:value:0*
T0*
_output_shapes
:@a
Identity_25IdentityIdentity_24:output:0"/device:CPU:0*
T0*
_output_shapes
:@k
Read_13/DisableCopyOnReadDisableCopyOnRead%read_13_disablecopyonread_variable_22*
_output_shapes
 �
Read_13/ReadVariableOpReadVariableOp%read_13_disablecopyonread_variable_22^Read_13/DisableCopyOnRead*
_output_shapes
:@*
dtype0\
Identity_26IdentityRead_13/ReadVariableOp:value:0*
T0*
_output_shapes
:@a
Identity_27IdentityIdentity_26:output:0"/device:CPU:0*
T0*
_output_shapes
:@k
Read_14/DisableCopyOnReadDisableCopyOnRead%read_14_disablecopyonread_variable_21*
_output_shapes
 �
Read_14/ReadVariableOpReadVariableOp%read_14_disablecopyonread_variable_21^Read_14/DisableCopyOnRead*
_output_shapes

:@*
dtype0`
Identity_28IdentityRead_14/ReadVariableOp:value:0*
T0*
_output_shapes

:@e
Identity_29IdentityIdentity_28:output:0"/device:CPU:0*
T0*
_output_shapes

:@k
Read_15/DisableCopyOnReadDisableCopyOnRead%read_15_disablecopyonread_variable_20*
_output_shapes
 �
Read_15/ReadVariableOpReadVariableOp%read_15_disablecopyonread_variable_20^Read_15/DisableCopyOnRead*
_output_shapes
:*
dtype0\
Identity_30IdentityRead_15/ReadVariableOp:value:0*
T0*
_output_shapes
:a
Identity_31IdentityIdentity_30:output:0"/device:CPU:0*
T0*
_output_shapes
:k
Read_16/DisableCopyOnReadDisableCopyOnRead%read_16_disablecopyonread_variable_19*
_output_shapes
 �
Read_16/ReadVariableOpReadVariableOp%read_16_disablecopyonread_variable_19^Read_16/DisableCopyOnRead*
_output_shapes
:	�*
dtype0a
Identity_32IdentityRead_16/ReadVariableOp:value:0*
T0*
_output_shapes
:	�f
Identity_33IdentityIdentity_32:output:0"/device:CPU:0*
T0*
_output_shapes
:	�k
Read_17/DisableCopyOnReadDisableCopyOnRead%read_17_disablecopyonread_variable_18*
_output_shapes
 �
Read_17/ReadVariableOpReadVariableOp%read_17_disablecopyonread_variable_18^Read_17/DisableCopyOnRead*
_output_shapes
:	�*
dtype0a
Identity_34IdentityRead_17/ReadVariableOp:value:0*
T0*
_output_shapes
:	�f
Identity_35IdentityIdentity_34:output:0"/device:CPU:0*
T0*
_output_shapes
:	�k
Read_18/DisableCopyOnReadDisableCopyOnRead%read_18_disablecopyonread_variable_17*
_output_shapes
 �
Read_18/ReadVariableOpReadVariableOp%read_18_disablecopyonread_variable_17^Read_18/DisableCopyOnRead*
_output_shapes	
:�*
dtype0]
Identity_36IdentityRead_18/ReadVariableOp:value:0*
T0*
_output_shapes	
:�b
Identity_37IdentityIdentity_36:output:0"/device:CPU:0*
T0*
_output_shapes	
:�k
Read_19/DisableCopyOnReadDisableCopyOnRead%read_19_disablecopyonread_variable_16*
_output_shapes
 �
Read_19/ReadVariableOpReadVariableOp%read_19_disablecopyonread_variable_16^Read_19/DisableCopyOnRead*
_output_shapes	
:�*
dtype0]
Identity_38IdentityRead_19/ReadVariableOp:value:0*
T0*
_output_shapes	
:�b
Identity_39IdentityIdentity_38:output:0"/device:CPU:0*
T0*
_output_shapes	
:�k
Read_20/DisableCopyOnReadDisableCopyOnRead%read_20_disablecopyonread_variable_15*
_output_shapes
 �
Read_20/ReadVariableOpReadVariableOp%read_20_disablecopyonread_variable_15^Read_20/DisableCopyOnRead*
_output_shapes	
:�*
dtype0]
Identity_40IdentityRead_20/ReadVariableOp:value:0*
T0*
_output_shapes	
:�b
Identity_41IdentityIdentity_40:output:0"/device:CPU:0*
T0*
_output_shapes	
:�k
Read_21/DisableCopyOnReadDisableCopyOnRead%read_21_disablecopyonread_variable_14*
_output_shapes
 �
Read_21/ReadVariableOpReadVariableOp%read_21_disablecopyonread_variable_14^Read_21/DisableCopyOnRead*
_output_shapes	
:�*
dtype0]
Identity_42IdentityRead_21/ReadVariableOp:value:0*
T0*
_output_shapes	
:�b
Identity_43IdentityIdentity_42:output:0"/device:CPU:0*
T0*
_output_shapes	
:�k
Read_22/DisableCopyOnReadDisableCopyOnRead%read_22_disablecopyonread_variable_13*
_output_shapes
 �
Read_22/ReadVariableOpReadVariableOp%read_22_disablecopyonread_variable_13^Read_22/DisableCopyOnRead*
_output_shapes	
:�*
dtype0]
Identity_44IdentityRead_22/ReadVariableOp:value:0*
T0*
_output_shapes	
:�b
Identity_45IdentityIdentity_44:output:0"/device:CPU:0*
T0*
_output_shapes	
:�k
Read_23/DisableCopyOnReadDisableCopyOnRead%read_23_disablecopyonread_variable_12*
_output_shapes
 �
Read_23/ReadVariableOpReadVariableOp%read_23_disablecopyonread_variable_12^Read_23/DisableCopyOnRead*
_output_shapes	
:�*
dtype0]
Identity_46IdentityRead_23/ReadVariableOp:value:0*
T0*
_output_shapes	
:�b
Identity_47IdentityIdentity_46:output:0"/device:CPU:0*
T0*
_output_shapes	
:�k
Read_24/DisableCopyOnReadDisableCopyOnRead%read_24_disablecopyonread_variable_11*
_output_shapes
 �
Read_24/ReadVariableOpReadVariableOp%read_24_disablecopyonread_variable_11^Read_24/DisableCopyOnRead*
_output_shapes
:	�@*
dtype0a
Identity_48IdentityRead_24/ReadVariableOp:value:0*
T0*
_output_shapes
:	�@f
Identity_49IdentityIdentity_48:output:0"/device:CPU:0*
T0*
_output_shapes
:	�@k
Read_25/DisableCopyOnReadDisableCopyOnRead%read_25_disablecopyonread_variable_10*
_output_shapes
 �
Read_25/ReadVariableOpReadVariableOp%read_25_disablecopyonread_variable_10^Read_25/DisableCopyOnRead*
_output_shapes
:	�@*
dtype0a
Identity_50IdentityRead_25/ReadVariableOp:value:0*
T0*
_output_shapes
:	�@f
Identity_51IdentityIdentity_50:output:0"/device:CPU:0*
T0*
_output_shapes
:	�@j
Read_26/DisableCopyOnReadDisableCopyOnRead$read_26_disablecopyonread_variable_9*
_output_shapes
 �
Read_26/ReadVariableOpReadVariableOp$read_26_disablecopyonread_variable_9^Read_26/DisableCopyOnRead*
_output_shapes
:@*
dtype0\
Identity_52IdentityRead_26/ReadVariableOp:value:0*
T0*
_output_shapes
:@a
Identity_53IdentityIdentity_52:output:0"/device:CPU:0*
T0*
_output_shapes
:@j
Read_27/DisableCopyOnReadDisableCopyOnRead$read_27_disablecopyonread_variable_8*
_output_shapes
 �
Read_27/ReadVariableOpReadVariableOp$read_27_disablecopyonread_variable_8^Read_27/DisableCopyOnRead*
_output_shapes
:@*
dtype0\
Identity_54IdentityRead_27/ReadVariableOp:value:0*
T0*
_output_shapes
:@a
Identity_55IdentityIdentity_54:output:0"/device:CPU:0*
T0*
_output_shapes
:@j
Read_28/DisableCopyOnReadDisableCopyOnRead$read_28_disablecopyonread_variable_7*
_output_shapes
 �
Read_28/ReadVariableOpReadVariableOp$read_28_disablecopyonread_variable_7^Read_28/DisableCopyOnRead*
_output_shapes
:@*
dtype0\
Identity_56IdentityRead_28/ReadVariableOp:value:0*
T0*
_output_shapes
:@a
Identity_57IdentityIdentity_56:output:0"/device:CPU:0*
T0*
_output_shapes
:@j
Read_29/DisableCopyOnReadDisableCopyOnRead$read_29_disablecopyonread_variable_6*
_output_shapes
 �
Read_29/ReadVariableOpReadVariableOp$read_29_disablecopyonread_variable_6^Read_29/DisableCopyOnRead*
_output_shapes
:@*
dtype0\
Identity_58IdentityRead_29/ReadVariableOp:value:0*
T0*
_output_shapes
:@a
Identity_59IdentityIdentity_58:output:0"/device:CPU:0*
T0*
_output_shapes
:@j
Read_30/DisableCopyOnReadDisableCopyOnRead$read_30_disablecopyonread_variable_5*
_output_shapes
 �
Read_30/ReadVariableOpReadVariableOp$read_30_disablecopyonread_variable_5^Read_30/DisableCopyOnRead*
_output_shapes
:@*
dtype0\
Identity_60IdentityRead_30/ReadVariableOp:value:0*
T0*
_output_shapes
:@a
Identity_61IdentityIdentity_60:output:0"/device:CPU:0*
T0*
_output_shapes
:@j
Read_31/DisableCopyOnReadDisableCopyOnRead$read_31_disablecopyonread_variable_4*
_output_shapes
 �
Read_31/ReadVariableOpReadVariableOp$read_31_disablecopyonread_variable_4^Read_31/DisableCopyOnRead*
_output_shapes
:@*
dtype0\
Identity_62IdentityRead_31/ReadVariableOp:value:0*
T0*
_output_shapes
:@a
Identity_63IdentityIdentity_62:output:0"/device:CPU:0*
T0*
_output_shapes
:@j
Read_32/DisableCopyOnReadDisableCopyOnRead$read_32_disablecopyonread_variable_3*
_output_shapes
 �
Read_32/ReadVariableOpReadVariableOp$read_32_disablecopyonread_variable_3^Read_32/DisableCopyOnRead*
_output_shapes

:@*
dtype0`
Identity_64IdentityRead_32/ReadVariableOp:value:0*
T0*
_output_shapes

:@e
Identity_65IdentityIdentity_64:output:0"/device:CPU:0*
T0*
_output_shapes

:@j
Read_33/DisableCopyOnReadDisableCopyOnRead$read_33_disablecopyonread_variable_2*
_output_shapes
 �
Read_33/ReadVariableOpReadVariableOp$read_33_disablecopyonread_variable_2^Read_33/DisableCopyOnRead*
_output_shapes

:@*
dtype0`
Identity_66IdentityRead_33/ReadVariableOp:value:0*
T0*
_output_shapes

:@e
Identity_67IdentityIdentity_66:output:0"/device:CPU:0*
T0*
_output_shapes

:@j
Read_34/DisableCopyOnReadDisableCopyOnRead$read_34_disablecopyonread_variable_1*
_output_shapes
 �
Read_34/ReadVariableOpReadVariableOp$read_34_disablecopyonread_variable_1^Read_34/DisableCopyOnRead*
_output_shapes
:*
dtype0\
Identity_68IdentityRead_34/ReadVariableOp:value:0*
T0*
_output_shapes
:a
Identity_69IdentityIdentity_68:output:0"/device:CPU:0*
T0*
_output_shapes
:h
Read_35/DisableCopyOnReadDisableCopyOnRead"read_35_disablecopyonread_variable*
_output_shapes
 �
Read_35/ReadVariableOpReadVariableOp"read_35_disablecopyonread_variable^Read_35/DisableCopyOnRead*
_output_shapes
:*
dtype0\
Identity_70IdentityRead_35/ReadVariableOp:value:0*
T0*
_output_shapes
:a
Identity_71IdentityIdentity_70:output:0"/device:CPU:0*
T0*
_output_shapes
:L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : �
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: �
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:%*
dtype0*�
value�B�%B0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB0_operations/5/_kernel/.ATTRIBUTES/VARIABLE_VALUEB-_operations/5/bias/.ATTRIBUTES/VARIABLE_VALUEB._operations/6/gamma/.ATTRIBUTES/VARIABLE_VALUEB-_operations/6/beta/.ATTRIBUTES/VARIABLE_VALUEB4_operations/6/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB8_operations/6/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB0_operations/8/_kernel/.ATTRIBUTES/VARIABLE_VALUEB-_operations/8/bias/.ATTRIBUTES/VARIABLE_VALUEB._operations/9/gamma/.ATTRIBUTES/VARIABLE_VALUEB-_operations/9/beta/.ATTRIBUTES/VARIABLE_VALUEB4_operations/9/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB8_operations/9/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB1_operations/11/_kernel/.ATTRIBUTES/VARIABLE_VALUEB._operations/11/bias/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/17/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/18/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/19/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/20/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/21/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:%*
dtype0*]
valueTBR%B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B �
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0Identity_1:output:0Identity_3:output:0Identity_5:output:0Identity_7:output:0Identity_9:output:0Identity_11:output:0Identity_13:output:0Identity_15:output:0Identity_17:output:0Identity_19:output:0Identity_21:output:0Identity_23:output:0Identity_25:output:0Identity_27:output:0Identity_29:output:0Identity_31:output:0Identity_33:output:0Identity_35:output:0Identity_37:output:0Identity_39:output:0Identity_41:output:0Identity_43:output:0Identity_45:output:0Identity_47:output:0Identity_49:output:0Identity_51:output:0Identity_53:output:0Identity_55:output:0Identity_57:output:0Identity_59:output:0Identity_61:output:0Identity_63:output:0Identity_65:output:0Identity_67:output:0Identity_69:output:0Identity_71:output:0savev2_const_4"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *3
dtypes)
'2%	�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 i
Identity_72Identityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: U
Identity_73IdentityIdentity_72:output:0^NoOp*
T0*
_output_shapes
: �
NoOpNoOp^MergeV2Checkpoints^Read/DisableCopyOnRead^Read/ReadVariableOp^Read_1/DisableCopyOnRead^Read_1/ReadVariableOp^Read_10/DisableCopyOnRead^Read_10/ReadVariableOp^Read_11/DisableCopyOnRead^Read_11/ReadVariableOp^Read_12/DisableCopyOnRead^Read_12/ReadVariableOp^Read_13/DisableCopyOnRead^Read_13/ReadVariableOp^Read_14/DisableCopyOnRead^Read_14/ReadVariableOp^Read_15/DisableCopyOnRead^Read_15/ReadVariableOp^Read_16/DisableCopyOnRead^Read_16/ReadVariableOp^Read_17/DisableCopyOnRead^Read_17/ReadVariableOp^Read_18/DisableCopyOnRead^Read_18/ReadVariableOp^Read_19/DisableCopyOnRead^Read_19/ReadVariableOp^Read_2/DisableCopyOnRead^Read_2/ReadVariableOp^Read_20/DisableCopyOnRead^Read_20/ReadVariableOp^Read_21/DisableCopyOnRead^Read_21/ReadVariableOp^Read_22/DisableCopyOnRead^Read_22/ReadVariableOp^Read_23/DisableCopyOnRead^Read_23/ReadVariableOp^Read_24/DisableCopyOnRead^Read_24/ReadVariableOp^Read_25/DisableCopyOnRead^Read_25/ReadVariableOp^Read_26/DisableCopyOnRead^Read_26/ReadVariableOp^Read_27/DisableCopyOnRead^Read_27/ReadVariableOp^Read_28/DisableCopyOnRead^Read_28/ReadVariableOp^Read_29/DisableCopyOnRead^Read_29/ReadVariableOp^Read_3/DisableCopyOnRead^Read_3/ReadVariableOp^Read_30/DisableCopyOnRead^Read_30/ReadVariableOp^Read_31/DisableCopyOnRead^Read_31/ReadVariableOp^Read_32/DisableCopyOnRead^Read_32/ReadVariableOp^Read_33/DisableCopyOnRead^Read_33/ReadVariableOp^Read_34/DisableCopyOnRead^Read_34/ReadVariableOp^Read_35/DisableCopyOnRead^Read_35/ReadVariableOp^Read_4/DisableCopyOnRead^Read_4/ReadVariableOp^Read_5/DisableCopyOnRead^Read_5/ReadVariableOp^Read_6/DisableCopyOnRead^Read_6/ReadVariableOp^Read_7/DisableCopyOnRead^Read_7/ReadVariableOp^Read_8/DisableCopyOnRead^Read_8/ReadVariableOp^Read_9/DisableCopyOnRead^Read_9/ReadVariableOp*
_output_shapes
 "#
identity_73Identity_73:output:0*(
_construction_contextkEagerRuntime*_
_input_shapesN
L: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2(
MergeV2CheckpointsMergeV2Checkpoints20
Read/DisableCopyOnReadRead/DisableCopyOnRead2*
Read/ReadVariableOpRead/ReadVariableOp24
Read_1/DisableCopyOnReadRead_1/DisableCopyOnRead2.
Read_1/ReadVariableOpRead_1/ReadVariableOp26
Read_10/DisableCopyOnReadRead_10/DisableCopyOnRead20
Read_10/ReadVariableOpRead_10/ReadVariableOp26
Read_11/DisableCopyOnReadRead_11/DisableCopyOnRead20
Read_11/ReadVariableOpRead_11/ReadVariableOp26
Read_12/DisableCopyOnReadRead_12/DisableCopyOnRead20
Read_12/ReadVariableOpRead_12/ReadVariableOp26
Read_13/DisableCopyOnReadRead_13/DisableCopyOnRead20
Read_13/ReadVariableOpRead_13/ReadVariableOp26
Read_14/DisableCopyOnReadRead_14/DisableCopyOnRead20
Read_14/ReadVariableOpRead_14/ReadVariableOp26
Read_15/DisableCopyOnReadRead_15/DisableCopyOnRead20
Read_15/ReadVariableOpRead_15/ReadVariableOp26
Read_16/DisableCopyOnReadRead_16/DisableCopyOnRead20
Read_16/ReadVariableOpRead_16/ReadVariableOp26
Read_17/DisableCopyOnReadRead_17/DisableCopyOnRead20
Read_17/ReadVariableOpRead_17/ReadVariableOp26
Read_18/DisableCopyOnReadRead_18/DisableCopyOnRead20
Read_18/ReadVariableOpRead_18/ReadVariableOp26
Read_19/DisableCopyOnReadRead_19/DisableCopyOnRead20
Read_19/ReadVariableOpRead_19/ReadVariableOp24
Read_2/DisableCopyOnReadRead_2/DisableCopyOnRead2.
Read_2/ReadVariableOpRead_2/ReadVariableOp26
Read_20/DisableCopyOnReadRead_20/DisableCopyOnRead20
Read_20/ReadVariableOpRead_20/ReadVariableOp26
Read_21/DisableCopyOnReadRead_21/DisableCopyOnRead20
Read_21/ReadVariableOpRead_21/ReadVariableOp26
Read_22/DisableCopyOnReadRead_22/DisableCopyOnRead20
Read_22/ReadVariableOpRead_22/ReadVariableOp26
Read_23/DisableCopyOnReadRead_23/DisableCopyOnRead20
Read_23/ReadVariableOpRead_23/ReadVariableOp26
Read_24/DisableCopyOnReadRead_24/DisableCopyOnRead20
Read_24/ReadVariableOpRead_24/ReadVariableOp26
Read_25/DisableCopyOnReadRead_25/DisableCopyOnRead20
Read_25/ReadVariableOpRead_25/ReadVariableOp26
Read_26/DisableCopyOnReadRead_26/DisableCopyOnRead20
Read_26/ReadVariableOpRead_26/ReadVariableOp26
Read_27/DisableCopyOnReadRead_27/DisableCopyOnRead20
Read_27/ReadVariableOpRead_27/ReadVariableOp26
Read_28/DisableCopyOnReadRead_28/DisableCopyOnRead20
Read_28/ReadVariableOpRead_28/ReadVariableOp26
Read_29/DisableCopyOnReadRead_29/DisableCopyOnRead20
Read_29/ReadVariableOpRead_29/ReadVariableOp24
Read_3/DisableCopyOnReadRead_3/DisableCopyOnRead2.
Read_3/ReadVariableOpRead_3/ReadVariableOp26
Read_30/DisableCopyOnReadRead_30/DisableCopyOnRead20
Read_30/ReadVariableOpRead_30/ReadVariableOp26
Read_31/DisableCopyOnReadRead_31/DisableCopyOnRead20
Read_31/ReadVariableOpRead_31/ReadVariableOp26
Read_32/DisableCopyOnReadRead_32/DisableCopyOnRead20
Read_32/ReadVariableOpRead_32/ReadVariableOp26
Read_33/DisableCopyOnReadRead_33/DisableCopyOnRead20
Read_33/ReadVariableOpRead_33/ReadVariableOp26
Read_34/DisableCopyOnReadRead_34/DisableCopyOnRead20
Read_34/ReadVariableOpRead_34/ReadVariableOp26
Read_35/DisableCopyOnReadRead_35/DisableCopyOnRead20
Read_35/ReadVariableOpRead_35/ReadVariableOp24
Read_4/DisableCopyOnReadRead_4/DisableCopyOnRead2.
Read_4/ReadVariableOpRead_4/ReadVariableOp24
Read_5/DisableCopyOnReadRead_5/DisableCopyOnRead2.
Read_5/ReadVariableOpRead_5/ReadVariableOp24
Read_6/DisableCopyOnReadRead_6/DisableCopyOnRead2.
Read_6/ReadVariableOpRead_6/ReadVariableOp24
Read_7/DisableCopyOnReadRead_7/DisableCopyOnRead2.
Read_7/ReadVariableOpRead_7/ReadVariableOp24
Read_8/DisableCopyOnReadRead_8/DisableCopyOnRead2.
Read_8/ReadVariableOpRead_8/ReadVariableOp24
Read_9/DisableCopyOnReadRead_9/DisableCopyOnRead2.
Read_9/ReadVariableOpRead_9/ReadVariableOp:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:+'
%
_user_specified_nameVariable_35:+'
%
_user_specified_nameVariable_34:+'
%
_user_specified_nameVariable_33:+'
%
_user_specified_nameVariable_32:+'
%
_user_specified_nameVariable_31:+'
%
_user_specified_nameVariable_30:+'
%
_user_specified_nameVariable_29:+'
%
_user_specified_nameVariable_28:+	'
%
_user_specified_nameVariable_27:+
'
%
_user_specified_nameVariable_26:+'
%
_user_specified_nameVariable_25:+'
%
_user_specified_nameVariable_24:+'
%
_user_specified_nameVariable_23:+'
%
_user_specified_nameVariable_22:+'
%
_user_specified_nameVariable_21:+'
%
_user_specified_nameVariable_20:+'
%
_user_specified_nameVariable_19:+'
%
_user_specified_nameVariable_18:+'
%
_user_specified_nameVariable_17:+'
%
_user_specified_nameVariable_16:+'
%
_user_specified_nameVariable_15:+'
%
_user_specified_nameVariable_14:+'
%
_user_specified_nameVariable_13:+'
%
_user_specified_nameVariable_12:+'
%
_user_specified_nameVariable_11:+'
%
_user_specified_nameVariable_10:*&
$
_user_specified_name
Variable_9:*&
$
_user_specified_name
Variable_8:*&
$
_user_specified_name
Variable_7:*&
$
_user_specified_name
Variable_6:*&
$
_user_specified_name
Variable_5:* &
$
_user_specified_name
Variable_4:*!&
$
_user_specified_name
Variable_3:*"&
$
_user_specified_name
Variable_2:*#&
$
_user_specified_name
Variable_1:($$
"
_user_specified_name
Variable:?%;

_output_shapes
: 
!
_user_specified_name	Const_4
��
�
&__inference_serve_tf_examples_fn_47892
examples"
transform_features_layer_47821"
transform_features_layer_47823"
transform_features_layer_47825"
transform_features_layer_47827D
1functional_1_dense_1_cast_readvariableop_resource:	�?
0functional_1_dense_1_add_readvariableop_resource:	�N
?functional_1_batch_normalization_1_cast_readvariableop_resource:	�P
Afunctional_1_batch_normalization_1_cast_1_readvariableop_resource:	�P
Afunctional_1_batch_normalization_1_cast_2_readvariableop_resource:	�P
Afunctional_1_batch_normalization_1_cast_3_readvariableop_resource:	�F
3functional_1_dense_1_2_cast_readvariableop_resource:	�@@
2functional_1_dense_1_2_add_readvariableop_resource:@O
Afunctional_1_batch_normalization_1_2_cast_readvariableop_resource:@Q
Cfunctional_1_batch_normalization_1_2_cast_1_readvariableop_resource:@Q
Cfunctional_1_batch_normalization_1_2_cast_2_readvariableop_resource:@Q
Cfunctional_1_batch_normalization_1_2_cast_3_readvariableop_resource:@E
3functional_1_dense_2_1_cast_readvariableop_resource:@@
2functional_1_dense_2_1_add_readvariableop_resource:
identity��6functional_1/batch_normalization_1/Cast/ReadVariableOp�8functional_1/batch_normalization_1/Cast_1/ReadVariableOp�8functional_1/batch_normalization_1/Cast_2/ReadVariableOp�8functional_1/batch_normalization_1/Cast_3/ReadVariableOp�8functional_1/batch_normalization_1_2/Cast/ReadVariableOp�:functional_1/batch_normalization_1_2/Cast_1/ReadVariableOp�:functional_1/batch_normalization_1_2/Cast_2/ReadVariableOp�:functional_1/batch_normalization_1_2/Cast_3/ReadVariableOp�'functional_1/dense_1/Add/ReadVariableOp�(functional_1/dense_1/Cast/ReadVariableOp�)functional_1/dense_1_2/Add/ReadVariableOp�*functional_1/dense_1_2/Cast/ReadVariableOp�)functional_1/dense_2_1/Add/ReadVariableOp�*functional_1/dense_2_1/Cast/ReadVariableOpU
ParseExample/ConstConst*
_output_shapes
: *
dtype0	*
valueB	 W
ParseExample/Const_1Const*
_output_shapes
: *
dtype0*
valueB W
ParseExample/Const_2Const*
_output_shapes
: *
dtype0	*
valueB	 W
ParseExample/Const_3Const*
_output_shapes
: *
dtype0	*
valueB	 d
!ParseExample/ParseExampleV2/namesConst*
_output_shapes
: *
dtype0*
valueB j
'ParseExample/ParseExampleV2/sparse_keysConst*
_output_shapes
: *
dtype0*
valueB �
&ParseExample/ParseExampleV2/dense_keysConst*
_output_shapes
:*
dtype0*<
value3B1BAgeBBMIBGenderBPhysicalActivityLevelj
'ParseExample/ParseExampleV2/ragged_keysConst*
_output_shapes
: *
dtype0*
valueB �
ParseExample/ParseExampleV2ParseExampleV2examples*ParseExample/ParseExampleV2/names:output:00ParseExample/ParseExampleV2/sparse_keys:output:0/ParseExample/ParseExampleV2/dense_keys:output:00ParseExample/ParseExampleV2/ragged_keys:output:0ParseExample/Const:output:0ParseExample/Const_1:output:0ParseExample/Const_2:output:0ParseExample/Const_3:output:0*
Tdense
2			*`
_output_shapesN
L:���������:���������:���������:���������**
dense_shapes
::::*

num_sparse *
ragged_split_types
 *
ragged_value_types
 *
sparse_types
 �
transform_features_layer/ShapeShape*ParseExample/ParseExampleV2:dense_values:0*
T0	*
_output_shapes
::��v
,transform_features_layer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: x
.transform_features_layer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:x
.transform_features_layer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
&transform_features_layer/strided_sliceStridedSlice'transform_features_layer/Shape:output:05transform_features_layer/strided_slice/stack:output:07transform_features_layer/strided_slice/stack_1:output:07transform_features_layer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
 transform_features_layer/Shape_1Shape*ParseExample/ParseExampleV2:dense_values:0*
T0	*
_output_shapes
::��x
.transform_features_layer/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: z
0transform_features_layer/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:z
0transform_features_layer/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
(transform_features_layer/strided_slice_1StridedSlice)transform_features_layer/Shape_1:output:07transform_features_layer/strided_slice_1/stack:output:09transform_features_layer/strided_slice_1/stack_1:output:09transform_features_layer/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maski
'transform_features_layer/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :�
%transform_features_layer/zeros/packedPack1transform_features_layer/strided_slice_1:output:00transform_features_layer/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:f
$transform_features_layer/zeros/ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R �
transform_features_layer/zerosFill.transform_features_layer/zeros/packed:output:0-transform_features_layer/zeros/Const:output:0*
T0	*'
_output_shapes
:����������
/transform_features_layer/PlaceholderWithDefaultPlaceholderWithDefault'transform_features_layer/zeros:output:0*'
_output_shapes
:���������*
dtype0	*
shape:����������
(transform_features_layer/PartitionedCallPartitionedCall*ParseExample/ParseExampleV2:dense_values:0*ParseExample/ParseExampleV2:dense_values:1*ParseExample/ParseExampleV2:dense_values:28transform_features_layer/PlaceholderWithDefault:output:0*ParseExample/ParseExampleV2:dense_values:3transform_features_layer_47821transform_features_layer_47823transform_features_layer_47825transform_features_layer_47827*
Tin
2					*
Tout	
2			*
_collective_manager_ids
 *s
_output_shapesa
_:���������:���������:���������:���������:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *!
fR
__inference_pruned_47747�
functional_1/CastCast1transform_features_layer/PartitionedCall:output:2*

DstT0*

SrcT0	*'
_output_shapes
:����������
functional_1/Cast_1Cast1transform_features_layer/PartitionedCall:output:4*

DstT0*

SrcT0	*'
_output_shapes
:���������q
&functional_1/concatenate_1/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
����������
!functional_1/concatenate_1/concatConcatV21transform_features_layer/PartitionedCall:output:01transform_features_layer/PartitionedCall:output:1functional_1/Cast:y:0functional_1/Cast_1:y:0/functional_1/concatenate_1/concat/axis:output:0*
N*
T0*'
_output_shapes
:����������
(functional_1/dense_1/Cast/ReadVariableOpReadVariableOp1functional_1_dense_1_cast_readvariableop_resource*
_output_shapes
:	�*
dtype0�
functional_1/dense_1/MatMulMatMul*functional_1/concatenate_1/concat:output:00functional_1/dense_1/Cast/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
'functional_1/dense_1/Add/ReadVariableOpReadVariableOp0functional_1_dense_1_add_readvariableop_resource*
_output_shapes	
:�*
dtype0�
functional_1/dense_1/AddAddV2%functional_1/dense_1/MatMul:product:0/functional_1/dense_1/Add/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������r
functional_1/dense_1/ReluRelufunctional_1/dense_1/Add:z:0*
T0*(
_output_shapes
:�����������
6functional_1/batch_normalization_1/Cast/ReadVariableOpReadVariableOp?functional_1_batch_normalization_1_cast_readvariableop_resource*
_output_shapes	
:�*
dtype0�
8functional_1/batch_normalization_1/Cast_1/ReadVariableOpReadVariableOpAfunctional_1_batch_normalization_1_cast_1_readvariableop_resource*
_output_shapes	
:�*
dtype0�
8functional_1/batch_normalization_1/Cast_2/ReadVariableOpReadVariableOpAfunctional_1_batch_normalization_1_cast_2_readvariableop_resource*
_output_shapes	
:�*
dtype0�
8functional_1/batch_normalization_1/Cast_3/ReadVariableOpReadVariableOpAfunctional_1_batch_normalization_1_cast_3_readvariableop_resource*
_output_shapes	
:�*
dtype0w
2functional_1/batch_normalization_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
0functional_1/batch_normalization_1/batchnorm/addAddV2@functional_1/batch_normalization_1/Cast_1/ReadVariableOp:value:0;functional_1/batch_normalization_1/batchnorm/add/y:output:0*
T0*
_output_shapes	
:��
2functional_1/batch_normalization_1/batchnorm/RsqrtRsqrt4functional_1/batch_normalization_1/batchnorm/add:z:0*
T0*
_output_shapes	
:��
0functional_1/batch_normalization_1/batchnorm/mulMul6functional_1/batch_normalization_1/batchnorm/Rsqrt:y:0@functional_1/batch_normalization_1/Cast_2/ReadVariableOp:value:0*
T0*
_output_shapes	
:��
2functional_1/batch_normalization_1/batchnorm/mul_1Mul'functional_1/dense_1/Relu:activations:04functional_1/batch_normalization_1/batchnorm/mul:z:0*
T0*(
_output_shapes
:�����������
2functional_1/batch_normalization_1/batchnorm/mul_2Mul>functional_1/batch_normalization_1/Cast/ReadVariableOp:value:04functional_1/batch_normalization_1/batchnorm/mul:z:0*
T0*
_output_shapes	
:��
0functional_1/batch_normalization_1/batchnorm/subSub@functional_1/batch_normalization_1/Cast_3/ReadVariableOp:value:06functional_1/batch_normalization_1/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:��
2functional_1/batch_normalization_1/batchnorm/add_1AddV26functional_1/batch_normalization_1/batchnorm/mul_1:z:04functional_1/batch_normalization_1/batchnorm/sub:z:0*
T0*(
_output_shapes
:�����������
*functional_1/dense_1_2/Cast/ReadVariableOpReadVariableOp3functional_1_dense_1_2_cast_readvariableop_resource*
_output_shapes
:	�@*
dtype0�
functional_1/dense_1_2/MatMulMatMul6functional_1/batch_normalization_1/batchnorm/add_1:z:02functional_1/dense_1_2/Cast/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
)functional_1/dense_1_2/Add/ReadVariableOpReadVariableOp2functional_1_dense_1_2_add_readvariableop_resource*
_output_shapes
:@*
dtype0�
functional_1/dense_1_2/AddAddV2'functional_1/dense_1_2/MatMul:product:01functional_1/dense_1_2/Add/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@u
functional_1/dense_1_2/ReluRelufunctional_1/dense_1_2/Add:z:0*
T0*'
_output_shapes
:���������@�
8functional_1/batch_normalization_1_2/Cast/ReadVariableOpReadVariableOpAfunctional_1_batch_normalization_1_2_cast_readvariableop_resource*
_output_shapes
:@*
dtype0�
:functional_1/batch_normalization_1_2/Cast_1/ReadVariableOpReadVariableOpCfunctional_1_batch_normalization_1_2_cast_1_readvariableop_resource*
_output_shapes
:@*
dtype0�
:functional_1/batch_normalization_1_2/Cast_2/ReadVariableOpReadVariableOpCfunctional_1_batch_normalization_1_2_cast_2_readvariableop_resource*
_output_shapes
:@*
dtype0�
:functional_1/batch_normalization_1_2/Cast_3/ReadVariableOpReadVariableOpCfunctional_1_batch_normalization_1_2_cast_3_readvariableop_resource*
_output_shapes
:@*
dtype0y
4functional_1/batch_normalization_1_2/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
2functional_1/batch_normalization_1_2/batchnorm/addAddV2Bfunctional_1/batch_normalization_1_2/Cast_1/ReadVariableOp:value:0=functional_1/batch_normalization_1_2/batchnorm/add/y:output:0*
T0*
_output_shapes
:@�
4functional_1/batch_normalization_1_2/batchnorm/RsqrtRsqrt6functional_1/batch_normalization_1_2/batchnorm/add:z:0*
T0*
_output_shapes
:@�
2functional_1/batch_normalization_1_2/batchnorm/mulMul8functional_1/batch_normalization_1_2/batchnorm/Rsqrt:y:0Bfunctional_1/batch_normalization_1_2/Cast_2/ReadVariableOp:value:0*
T0*
_output_shapes
:@�
4functional_1/batch_normalization_1_2/batchnorm/mul_1Mul)functional_1/dense_1_2/Relu:activations:06functional_1/batch_normalization_1_2/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������@�
4functional_1/batch_normalization_1_2/batchnorm/mul_2Mul@functional_1/batch_normalization_1_2/Cast/ReadVariableOp:value:06functional_1/batch_normalization_1_2/batchnorm/mul:z:0*
T0*
_output_shapes
:@�
2functional_1/batch_normalization_1_2/batchnorm/subSubBfunctional_1/batch_normalization_1_2/Cast_3/ReadVariableOp:value:08functional_1/batch_normalization_1_2/batchnorm/mul_2:z:0*
T0*
_output_shapes
:@�
4functional_1/batch_normalization_1_2/batchnorm/add_1AddV28functional_1/batch_normalization_1_2/batchnorm/mul_1:z:06functional_1/batch_normalization_1_2/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������@�
*functional_1/dense_2_1/Cast/ReadVariableOpReadVariableOp3functional_1_dense_2_1_cast_readvariableop_resource*
_output_shapes

:@*
dtype0�
functional_1/dense_2_1/MatMulMatMul8functional_1/batch_normalization_1_2/batchnorm/add_1:z:02functional_1/dense_2_1/Cast/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
)functional_1/dense_2_1/Add/ReadVariableOpReadVariableOp2functional_1_dense_2_1_add_readvariableop_resource*
_output_shapes
:*
dtype0�
functional_1/dense_2_1/AddAddV2'functional_1/dense_2_1/MatMul:product:01functional_1/dense_2_1/Add/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������{
functional_1/dense_2_1/SoftmaxSoftmaxfunctional_1/dense_2_1/Add:z:0*
T0*'
_output_shapes
:���������w
IdentityIdentity(functional_1/dense_2_1/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp7^functional_1/batch_normalization_1/Cast/ReadVariableOp9^functional_1/batch_normalization_1/Cast_1/ReadVariableOp9^functional_1/batch_normalization_1/Cast_2/ReadVariableOp9^functional_1/batch_normalization_1/Cast_3/ReadVariableOp9^functional_1/batch_normalization_1_2/Cast/ReadVariableOp;^functional_1/batch_normalization_1_2/Cast_1/ReadVariableOp;^functional_1/batch_normalization_1_2/Cast_2/ReadVariableOp;^functional_1/batch_normalization_1_2/Cast_3/ReadVariableOp(^functional_1/dense_1/Add/ReadVariableOp)^functional_1/dense_1/Cast/ReadVariableOp*^functional_1/dense_1_2/Add/ReadVariableOp+^functional_1/dense_1_2/Cast/ReadVariableOp*^functional_1/dense_2_1/Add/ReadVariableOp+^functional_1/dense_2_1/Cast/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:���������: : : : : : : : : : : : : : : : : : 2p
6functional_1/batch_normalization_1/Cast/ReadVariableOp6functional_1/batch_normalization_1/Cast/ReadVariableOp2t
8functional_1/batch_normalization_1/Cast_1/ReadVariableOp8functional_1/batch_normalization_1/Cast_1/ReadVariableOp2t
8functional_1/batch_normalization_1/Cast_2/ReadVariableOp8functional_1/batch_normalization_1/Cast_2/ReadVariableOp2t
8functional_1/batch_normalization_1/Cast_3/ReadVariableOp8functional_1/batch_normalization_1/Cast_3/ReadVariableOp2t
8functional_1/batch_normalization_1_2/Cast/ReadVariableOp8functional_1/batch_normalization_1_2/Cast/ReadVariableOp2x
:functional_1/batch_normalization_1_2/Cast_1/ReadVariableOp:functional_1/batch_normalization_1_2/Cast_1/ReadVariableOp2x
:functional_1/batch_normalization_1_2/Cast_2/ReadVariableOp:functional_1/batch_normalization_1_2/Cast_2/ReadVariableOp2x
:functional_1/batch_normalization_1_2/Cast_3/ReadVariableOp:functional_1/batch_normalization_1_2/Cast_3/ReadVariableOp2R
'functional_1/dense_1/Add/ReadVariableOp'functional_1/dense_1/Add/ReadVariableOp2T
(functional_1/dense_1/Cast/ReadVariableOp(functional_1/dense_1/Cast/ReadVariableOp2V
)functional_1/dense_1_2/Add/ReadVariableOp)functional_1/dense_1_2/Add/ReadVariableOp2X
*functional_1/dense_1_2/Cast/ReadVariableOp*functional_1/dense_1_2/Cast/ReadVariableOp2V
)functional_1/dense_2_1/Add/ReadVariableOp)functional_1/dense_2_1/Add/ReadVariableOp2X
*functional_1/dense_2_1/Cast/ReadVariableOp*functional_1/dense_2_1/Cast/ReadVariableOp:M I
#
_output_shapes
:���������
"
_user_specified_name
examples:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:(	$
"
_user_specified_name
resource:(
$
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�
�
#__inference_signature_wrapper_47934
examples
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3:	�
	unknown_4:	�
	unknown_5:	�
	unknown_6:	�
	unknown_7:	�
	unknown_8:	�
	unknown_9:	�@

unknown_10:@

unknown_11:@

unknown_12:@

unknown_13:@

unknown_14:@

unknown_15:@

unknown_16:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallexamplesunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*0
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� */
f*R(
&__inference_serve_tf_examples_fn_47892o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:���������: : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:M I
#
_output_shapes
:���������
"
_user_specified_name
examples:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :%!

_user_specified_name47904:%!

_user_specified_name47906:%!

_user_specified_name47908:%!

_user_specified_name47910:%	!

_user_specified_name47912:%
!

_user_specified_name47914:%!

_user_specified_name47916:%!

_user_specified_name47918:%!

_user_specified_name47920:%!

_user_specified_name47922:%!

_user_specified_name47924:%!

_user_specified_name47926:%!

_user_specified_name47928:%!

_user_specified_name47930
�#
�
 __inference__wrapped_model_48034
age	
bmi

gender	
physicalactivitylevel	"
transform_features_layer_48017"
transform_features_layer_48019"
transform_features_layer_48021"
transform_features_layer_48023
identity

identity_1

identity_2	

identity_3	_
transform_features_layer/ShapeShapeage*
T0	*
_output_shapes
::��v
,transform_features_layer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: x
.transform_features_layer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:x
.transform_features_layer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
&transform_features_layer/strided_sliceStridedSlice'transform_features_layer/Shape:output:05transform_features_layer/strided_slice/stack:output:07transform_features_layer/strided_slice/stack_1:output:07transform_features_layer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maska
 transform_features_layer/Shape_1Shapeage*
T0	*
_output_shapes
::��x
.transform_features_layer/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: z
0transform_features_layer/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:z
0transform_features_layer/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
(transform_features_layer/strided_slice_1StridedSlice)transform_features_layer/Shape_1:output:07transform_features_layer/strided_slice_1/stack:output:09transform_features_layer/strided_slice_1/stack_1:output:09transform_features_layer/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maski
'transform_features_layer/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :�
%transform_features_layer/zeros/packedPack1transform_features_layer/strided_slice_1:output:00transform_features_layer/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:f
$transform_features_layer/zeros/ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R �
transform_features_layer/zerosFill.transform_features_layer/zeros/packed:output:0-transform_features_layer/zeros/Const:output:0*
T0	*'
_output_shapes
:����������
/transform_features_layer/PlaceholderWithDefaultPlaceholderWithDefault'transform_features_layer/zeros:output:0*'
_output_shapes
:���������*
dtype0	*
shape:����������
(transform_features_layer/PartitionedCallPartitionedCallagebmigender8transform_features_layer/PlaceholderWithDefault:output:0physicalactivityleveltransform_features_layer_48017transform_features_layer_48019transform_features_layer_48021transform_features_layer_48023*
Tin
2					*
Tout	
2			*
_collective_manager_ids
 *s
_output_shapesa
_:���������:���������:���������:���������:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *!
fR
__inference_pruned_47747y
IdentityIdentity1transform_features_layer/PartitionedCall:output:0*
T0*'
_output_shapes
:���������{

Identity_1Identity1transform_features_layer/PartitionedCall:output:1*
T0*'
_output_shapes
:���������{

Identity_2Identity1transform_features_layer/PartitionedCall:output:2*
T0	*'
_output_shapes
:���������{

Identity_3Identity1transform_features_layer/PartitionedCall:output:4*
T0	*'
_output_shapes
:���������"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*(
_construction_contextkEagerRuntime*g
_input_shapesV
T:���������:���������:���������:���������: : : : :L H
'
_output_shapes
:���������

_user_specified_nameAge:LH
'
_output_shapes
:���������

_user_specified_nameBMI:OK
'
_output_shapes
:���������
 
_user_specified_nameGender:^Z
'
_output_shapes
:���������
/
_user_specified_namePhysicalActivityLevel:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
��
�
!__inference__traced_restore_48598
file_prefix&
assignvariableop_variable_35:	 (
assignvariableop_1_variable_34: 1
assignvariableop_2_variable_33:	�-
assignvariableop_3_variable_32:	�-
assignvariableop_4_variable_31:	�-
assignvariableop_5_variable_30:	�-
assignvariableop_6_variable_29:	�-
assignvariableop_7_variable_28:	�1
assignvariableop_8_variable_27:	�@,
assignvariableop_9_variable_26:@-
assignvariableop_10_variable_25:@-
assignvariableop_11_variable_24:@-
assignvariableop_12_variable_23:@-
assignvariableop_13_variable_22:@1
assignvariableop_14_variable_21:@-
assignvariableop_15_variable_20:2
assignvariableop_16_variable_19:	�2
assignvariableop_17_variable_18:	�.
assignvariableop_18_variable_17:	�.
assignvariableop_19_variable_16:	�.
assignvariableop_20_variable_15:	�.
assignvariableop_21_variable_14:	�.
assignvariableop_22_variable_13:	�.
assignvariableop_23_variable_12:	�2
assignvariableop_24_variable_11:	�@2
assignvariableop_25_variable_10:	�@,
assignvariableop_26_variable_9:@,
assignvariableop_27_variable_8:@,
assignvariableop_28_variable_7:@,
assignvariableop_29_variable_6:@,
assignvariableop_30_variable_5:@,
assignvariableop_31_variable_4:@0
assignvariableop_32_variable_3:@0
assignvariableop_33_variable_2:@,
assignvariableop_34_variable_1:*
assignvariableop_35_variable:
identity_37��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_18�AssignVariableOp_19�AssignVariableOp_2�AssignVariableOp_20�AssignVariableOp_21�AssignVariableOp_22�AssignVariableOp_23�AssignVariableOp_24�AssignVariableOp_25�AssignVariableOp_26�AssignVariableOp_27�AssignVariableOp_28�AssignVariableOp_29�AssignVariableOp_3�AssignVariableOp_30�AssignVariableOp_31�AssignVariableOp_32�AssignVariableOp_33�AssignVariableOp_34�AssignVariableOp_35�AssignVariableOp_4�AssignVariableOp_5�AssignVariableOp_6�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:%*
dtype0*�
value�B�%B0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB0_operations/5/_kernel/.ATTRIBUTES/VARIABLE_VALUEB-_operations/5/bias/.ATTRIBUTES/VARIABLE_VALUEB._operations/6/gamma/.ATTRIBUTES/VARIABLE_VALUEB-_operations/6/beta/.ATTRIBUTES/VARIABLE_VALUEB4_operations/6/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB8_operations/6/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB0_operations/8/_kernel/.ATTRIBUTES/VARIABLE_VALUEB-_operations/8/bias/.ATTRIBUTES/VARIABLE_VALUEB._operations/9/gamma/.ATTRIBUTES/VARIABLE_VALUEB-_operations/9/beta/.ATTRIBUTES/VARIABLE_VALUEB4_operations/9/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB8_operations/9/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB1_operations/11/_kernel/.ATTRIBUTES/VARIABLE_VALUEB._operations/11/bias/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/17/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/18/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/19/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/20/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/21/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:%*
dtype0*]
valueTBR%B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B �
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*�
_output_shapes�
�:::::::::::::::::::::::::::::::::::::*3
dtypes)
'2%	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0	*
_output_shapes
:�
AssignVariableOpAssignVariableOpassignvariableop_variable_35Identity:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0	]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_1AssignVariableOpassignvariableop_1_variable_34Identity_1:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_2AssignVariableOpassignvariableop_2_variable_33Identity_2:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_3AssignVariableOpassignvariableop_3_variable_32Identity_3:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_4AssignVariableOpassignvariableop_4_variable_31Identity_4:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_5AssignVariableOpassignvariableop_5_variable_30Identity_5:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_6AssignVariableOpassignvariableop_6_variable_29Identity_6:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_7AssignVariableOpassignvariableop_7_variable_28Identity_7:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_8AssignVariableOpassignvariableop_8_variable_27Identity_8:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_9AssignVariableOpassignvariableop_9_variable_26Identity_9:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_10AssignVariableOpassignvariableop_10_variable_25Identity_10:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_11AssignVariableOpassignvariableop_11_variable_24Identity_11:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_12AssignVariableOpassignvariableop_12_variable_23Identity_12:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_13AssignVariableOpassignvariableop_13_variable_22Identity_13:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_14AssignVariableOpassignvariableop_14_variable_21Identity_14:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_15AssignVariableOpassignvariableop_15_variable_20Identity_15:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_16AssignVariableOpassignvariableop_16_variable_19Identity_16:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_17AssignVariableOpassignvariableop_17_variable_18Identity_17:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_18AssignVariableOpassignvariableop_18_variable_17Identity_18:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_19AssignVariableOpassignvariableop_19_variable_16Identity_19:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_20AssignVariableOpassignvariableop_20_variable_15Identity_20:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_21AssignVariableOpassignvariableop_21_variable_14Identity_21:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_22AssignVariableOpassignvariableop_22_variable_13Identity_22:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_23AssignVariableOpassignvariableop_23_variable_12Identity_23:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_24AssignVariableOpassignvariableop_24_variable_11Identity_24:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_25AssignVariableOpassignvariableop_25_variable_10Identity_25:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_26AssignVariableOpassignvariableop_26_variable_9Identity_26:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_27AssignVariableOpassignvariableop_27_variable_8Identity_27:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_28AssignVariableOpassignvariableop_28_variable_7Identity_28:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_29AssignVariableOpassignvariableop_29_variable_6Identity_29:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_30AssignVariableOpassignvariableop_30_variable_5Identity_30:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_31AssignVariableOpassignvariableop_31_variable_4Identity_31:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_32AssignVariableOpassignvariableop_32_variable_3Identity_32:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_33AssignVariableOpassignvariableop_33_variable_2Identity_33:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_34AssignVariableOpassignvariableop_34_variable_1Identity_34:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_35AssignVariableOpassignvariableop_35_variableIdentity_35:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0Y
NoOpNoOp"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 �
Identity_36Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_37IdentityIdentity_36:output:0^NoOp_1*
T0*
_output_shapes
: �
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
_output_shapes
 "#
identity_37Identity_37:output:0*(
_construction_contextkEagerRuntime*]
_input_shapesL
J: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:+'
%
_user_specified_nameVariable_35:+'
%
_user_specified_nameVariable_34:+'
%
_user_specified_nameVariable_33:+'
%
_user_specified_nameVariable_32:+'
%
_user_specified_nameVariable_31:+'
%
_user_specified_nameVariable_30:+'
%
_user_specified_nameVariable_29:+'
%
_user_specified_nameVariable_28:+	'
%
_user_specified_nameVariable_27:+
'
%
_user_specified_nameVariable_26:+'
%
_user_specified_nameVariable_25:+'
%
_user_specified_nameVariable_24:+'
%
_user_specified_nameVariable_23:+'
%
_user_specified_nameVariable_22:+'
%
_user_specified_nameVariable_21:+'
%
_user_specified_nameVariable_20:+'
%
_user_specified_nameVariable_19:+'
%
_user_specified_nameVariable_18:+'
%
_user_specified_nameVariable_17:+'
%
_user_specified_nameVariable_16:+'
%
_user_specified_nameVariable_15:+'
%
_user_specified_nameVariable_14:+'
%
_user_specified_nameVariable_13:+'
%
_user_specified_nameVariable_12:+'
%
_user_specified_nameVariable_11:+'
%
_user_specified_nameVariable_10:*&
$
_user_specified_name
Variable_9:*&
$
_user_specified_name
Variable_8:*&
$
_user_specified_name
Variable_7:*&
$
_user_specified_name
Variable_6:*&
$
_user_specified_name
Variable_5:* &
$
_user_specified_name
Variable_4:*!&
$
_user_specified_name
Variable_3:*"&
$
_user_specified_name
Variable_2:*#&
$
_user_specified_name
Variable_1:($$
"
_user_specified_name
Variable"�L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
9
examples-
serving_default_examples:0���������<
output_00
StatefulPartitionedCall:0���������tensorflow/serving/predict:�y
�
_tracked
_inbound_nodes
_outbound_nodes
_losses
_losses_override
_operations
_layers
_build_shapes_dict
	output_names

	optimizer
	tft_layer
_default_save_signature

signatures"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
v
0
1
2
3
4
5
6
7
8
9
10
11"
trackable_list_wrapper
v
0
1
2
3
4
5
6
7
8
9
10
11"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
�

_variables
_trainable_variables
 _trainable_variables_indices
_iterations
_learning_rate

_momentums
 _velocities"
_generic_user_object
�
!	variables
"trainable_variables
#regularization_losses
$	keras_api
%__call__
*&&call_and_return_all_conditional_losses
'_default_save_signature
$( _saved_model_loader_tracked_dict"
_tf_keras_model
�
)trace_02�
!__inference_serving_default_47996�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *p�m
k�h
����������
����������
����������
����������z)trace_0
,
*serving_default"
signature_map
y
+_inbound_nodes
,_outbound_nodes
-_losses
.	_loss_ids
/_losses_override"
_generic_user_object
y
0_inbound_nodes
1_outbound_nodes
2_losses
3	_loss_ids
4_losses_override"
_generic_user_object
y
5_inbound_nodes
6_outbound_nodes
7_losses
8	_loss_ids
9_losses_override"
_generic_user_object
y
:_inbound_nodes
;_outbound_nodes
<_losses
=	_loss_ids
>_losses_override"
_generic_user_object
�
?_inbound_nodes
@_outbound_nodes
A_losses
B	_loss_ids
C_losses_override
D_build_shapes_dict"
_generic_user_object
�
E_kernel
Fbias
G_inbound_nodes
H_outbound_nodes
I_losses
J	_loss_ids
K_losses_override
L_build_shapes_dict"
_generic_user_object
�
	Mgamma
Nbeta
Omoving_mean
Pmoving_variance
Q_inbound_nodes
R_outbound_nodes
S_losses
T	_loss_ids
U_losses_override
V_reduction_axes
W_build_shapes_dict"
_generic_user_object
�
X_inbound_nodes
Y_outbound_nodes
Z_losses
[	_loss_ids
\_losses_override
]_build_shapes_dict"
_generic_user_object
�
^_kernel
_bias
`_inbound_nodes
a_outbound_nodes
b_losses
c	_loss_ids
d_losses_override
e_build_shapes_dict"
_generic_user_object
�
	fgamma
gbeta
hmoving_mean
imoving_variance
j_inbound_nodes
k_outbound_nodes
l_losses
m	_loss_ids
n_losses_override
o_reduction_axes
p_build_shapes_dict"
_generic_user_object
�
q_inbound_nodes
r_outbound_nodes
s_losses
t	_loss_ids
u_losses_override
v_build_shapes_dict"
_generic_user_object
�
w_kernel
xbias
y_inbound_nodes
z_outbound_nodes
{_losses
|	_loss_ids
}_losses_override
~_build_shapes_dict"
_generic_user_object
�
0
1
2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13
�14
�15
�16
�17
�18
�19
�20
�21"
trackable_list_wrapper
f
E0
F1
M2
N3
^4
_5
f6
g7
w8
x9"
trackable_list_wrapper
 "
trackable_dict_wrapper
:	 2adam/iteration
: 2adam/learning_rate
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
!	variables
"trainable_variables
#regularization_losses
%__call__
'_default_save_signature
*&&call_and_return_all_conditional_losses
&&"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
8__inference_transform_features_layer_layer_call_fn_48095�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
S__inference_transform_features_layer_layer_call_and_return_conditional_losses_48073�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�	capture_0
�	capture_1
�	capture_2
�	capture_3B�
 __inference__wrapped_model_48034AgeBMIGenderPhysicalActivityLevel"�
���
FullArgSpec
args�

jargs_0
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�	capture_0z�	capture_1z�	capture_2z�	capture_3
�
�	_imported
�_wrapped_function
�_structured_inputs
�_structured_outputs
�_output_to_inputs_map"
trackable_dict_wrapper
�B�
!__inference_serving_default_47996inputsinputs_1inputs_2inputs_3"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�
�	capture_0
�	capture_1
�	capture_2
�	capture_3B�
#__inference_signature_wrapper_47934examples"�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs�

jexamples
kwonlydefaults
 
annotations� *
 z�	capture_0z�	capture_1z�	capture_2z�	capture_3
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
:	�2dense/kernel
:�2
dense/bias
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
(:&�2batch_normalization/gamma
':%�2batch_normalization/beta
,:*�2batch_normalization/moving_mean
0:.�2#batch_normalization/moving_variance
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
!:	�@2dense_1/kernel
:@2dense_1/bias
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
):'@2batch_normalization_1/gamma
(:&@2batch_normalization_1/beta
-:+@2!batch_normalization_1/moving_mean
1:/@2%batch_normalization_1/moving_variance
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 :@2dense_2/kernel
:2dense_2/bias
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
+:)	�2adam/dense_kernel_momentum
+:)	�2adam/dense_kernel_velocity
%:#�2adam/dense_bias_momentum
%:#�2adam/dense_bias_velocity
4:2�2'adam/batch_normalization_gamma_momentum
4:2�2'adam/batch_normalization_gamma_velocity
3:1�2&adam/batch_normalization_beta_momentum
3:1�2&adam/batch_normalization_beta_velocity
-:+	�@2adam/dense_1_kernel_momentum
-:+	�@2adam/dense_1_kernel_velocity
&:$@2adam/dense_1_bias_momentum
&:$@2adam/dense_1_bias_velocity
5:3@2)adam/batch_normalization_1_gamma_momentum
5:3@2)adam/batch_normalization_1_gamma_velocity
4:2@2(adam/batch_normalization_1_beta_momentum
4:2@2(adam/batch_normalization_1_beta_velocity
,:*@2adam/dense_2_kernel_momentum
,:*@2adam/dense_2_kernel_velocity
&:$2adam/dense_2_bias_momentum
&:$2adam/dense_2_bias_velocity
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�
�	capture_0
�	capture_1
�	capture_2
�	capture_3B�
8__inference_transform_features_layer_layer_call_fn_48095AgeBMIGenderPhysicalActivityLevel"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�	capture_0z�	capture_1z�	capture_2z�	capture_3
�
�	capture_0
�	capture_1
�	capture_2
�	capture_3B�
S__inference_transform_features_layer_layer_call_and_return_conditional_losses_48073AgeBMIGenderPhysicalActivityLevel"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�	capture_0z�	capture_1z�	capture_2z�	capture_3
!J	
Const_3jtf.TrackableConstant
!J	
Const_2jtf.TrackableConstant
!J	
Const_1jtf.TrackableConstant
J
Constjtf.TrackableConstant
�
�created_variables
�	resources
�trackable_objects
�initializers
�assets
�
signatures
$�_self_saveable_object_factories
�transform_fn"
_generic_user_object
�
�	capture_0
�	capture_1
�	capture_2
�	capture_3B�
__inference_pruned_47747inputsinputs_1inputs_2inputs_3inputs_4"�
���
FullArgSpec
args�	
jarg_0
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�	capture_0z�	capture_1z�	capture_2z�	capture_3
 "
trackable_dict_wrapper
 "
trackable_dict_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
-
�serving_default"
signature_map
 "
trackable_dict_wrapper
�
�	capture_0
�	capture_1
�	capture_2
�	capture_3B�
#__inference_signature_wrapper_47768inputsinputs_1inputs_2inputs_3inputs_4"�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 K

kwonlyargs=�:
jinputs

jinputs_1

jinputs_2

jinputs_3

jinputs_4
kwonlydefaults
 
annotations� *
 z�	capture_0z�	capture_1z�	capture_2z�	capture_3�
 __inference__wrapped_model_48034��������
���
���
$
Age�
Age���������	
$
BMI�
BMI���������
*
Gender �
Gender���������	
H
PhysicalActivityLevel/�,
PhysicalActivityLevel���������	
� "���
*
age_xf �
age_xf���������
*
bmi_xf �
bmi_xf���������
0
	gender_xf#� 
	gender_xf���������	
N
physicalactivitylevel_xf2�/
physicalactivitylevel_xf���������	�
__inference_pruned_47747��������
���
���
+
Age$�!

inputs_age���������	
+
BMI$�!

inputs_bmi���������
1
Gender'�$
inputs_gender���������	
C
ObesityCategory0�-
inputs_obesitycategory���������	
O
PhysicalActivityLevel6�3
inputs_physicalactivitylevel���������	
� "���
*
age_xf �
age_xf���������
*
bmi_xf �
bmi_xf���������
0
	gender_xf#� 
	gender_xf���������	
B
obesitycategory_xf,�)
obesitycategory_xf���������	
N
physicalactivitylevel_xf2�/
physicalactivitylevel_xf���������	�
!__inference_serving_default_47996�EFOPMN^_hifgwx���
���
���
"�
inputs_0���������
"�
inputs_1���������
"�
inputs_2���������
"�
inputs_3���������
� "!�
unknown����������
#__inference_signature_wrapper_47768��������
� 
���
*
inputs �
inputs���������	
.
inputs_1"�
inputs_1���������
.
inputs_2"�
inputs_2���������	
.
inputs_3"�
inputs_3���������	
.
inputs_4"�
inputs_4���������	"���
*
age_xf �
age_xf���������
*
bmi_xf �
bmi_xf���������
0
	gender_xf#� 
	gender_xf���������	
B
obesitycategory_xf,�)
obesitycategory_xf���������	
N
physicalactivitylevel_xf2�/
physicalactivitylevel_xf���������	�
#__inference_signature_wrapper_47934�����EFOPMN^_hifgwx9�6
� 
/�,
*
examples�
examples���������"3�0
.
output_0"�
output_0����������
S__inference_transform_features_layer_layer_call_and_return_conditional_losses_48073��������
���
���
$
Age�
Age���������	
$
BMI�
BMI���������
*
Gender �
Gender���������	
H
PhysicalActivityLevel/�,
PhysicalActivityLevel���������	
� "���
���
3
age_xf)�&
tensor_0_age_xf���������
3
bmi_xf)�&
tensor_0_bmi_xf���������
9
	gender_xf,�)
tensor_0_gender_xf���������	
W
physicalactivitylevel_xf;�8
!tensor_0_physicalactivitylevel_xf���������	
� �
8__inference_transform_features_layer_layer_call_fn_48095��������
���
���
$
Age�
Age���������	
$
BMI�
BMI���������
*
Gender �
Gender���������	
H
PhysicalActivityLevel/�,
PhysicalActivityLevel���������	
� "���
*
age_xf �
age_xf���������
*
bmi_xf �
bmi_xf���������
0
	gender_xf#� 
	gender_xf���������	
N
physicalactivitylevel_xf2�/
physicalactivitylevel_xf���������	