×
ˇ
D
AddV2
x"T
y"T
z"T"
Ttype:
2	
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
8
Const
output"dtype"
valuetensor"
dtypetype
.
Identity

input"T
output"T"	
Ttype

MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( 

NoOp
U
NotEqual
x"T
y"T
z
"	
Ttype"$
incompatible_shape_errorbool(
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
ł
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
@
RealDiv
x"T
y"T
z"T"
Ttype:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
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
H
ShardedFilename
basename	
shard

num_shards
filename
-
Sqrt
x"T
y"T"
Ttype:

2
Á
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
executor_typestring ¨
@
StaticRegexFullMatch	
input

output
"
patternstring
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
&
	ZerosLike
x"T
y"T"	
Ttype"serve*2.16.22v2.16.1-19-g810f233968c8a
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
 *ĘĆA
L
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *7*¤C
L
Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *ëąFB
y
serving_default_inputsPlaceholder*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0	*
shape:˙˙˙˙˙˙˙˙˙
{
serving_default_inputs_1Placeholder*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0*
shape:˙˙˙˙˙˙˙˙˙
{
serving_default_inputs_2Placeholder*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0	*
shape:˙˙˙˙˙˙˙˙˙
{
serving_default_inputs_3Placeholder*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0	*
shape:˙˙˙˙˙˙˙˙˙
{
serving_default_inputs_4Placeholder*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0	*
shape:˙˙˙˙˙˙˙˙˙
˙
PartitionedCallPartitionedCallserving_default_inputsserving_default_inputs_1serving_default_inputs_2serving_default_inputs_3serving_default_inputs_4Const_3Const_2Const_1Const*
Tin
2					*
Tout	
2			*
_collective_manager_ids
 *s
_output_shapesa
_:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *+
f&R$
"__inference_signature_wrapper_1239

NoOpNoOp

Const_4Const"/device:CPU:0*
_output_shapes
: *
dtype0*Ň
valueČBĹ Bž

created_variables
	resources
trackable_objects
initializers

assets
transform_fn

signatures* 
* 
* 
* 
* 
* 
>
	capture_0
		capture_1

	capture_2
	capture_3* 

serving_default* 
* 
* 
* 
* 
>
	capture_0
		capture_1

	capture_2
	capture_3* 
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 

StatefulPartitionedCallStatefulPartitionedCallsaver_filenameConst_4*
Tin
2*
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
GPU 2J 8 *&
f!R
__inference__traced_save_1273

StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename*
Tin
2*
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
GPU 2J 8 *)
f$R"
 __inference__traced_restore_1282źM

F
 __inference__traced_restore_1282
file_prefix

identity_1
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPHr
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B Ł
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*
_output_shapes
:*
dtypes
2Y
NoOpNoOp"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 X
IdentityIdentityfile_prefix^NoOp"/device:CPU:0*
T0*
_output_shapes
: J

Identity_1IdentityIdentity:output:0*
T0*
_output_shapes
: "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: :C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
ú
đ
"__inference_signature_wrapper_1239

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

identity_4	Ź
PartitionedCallPartitionedCallinputsinputs_1inputs_2inputs_3inputs_4unknown	unknown_0	unknown_1	unknown_2*
Tin
2					*
Tout	
2			*
_collective_manager_ids
 *s
_output_shapesa
_:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 * 
fR
__inference_pruned_1213`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙b

Identity_1IdentityPartitionedCall:output:1*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙b

Identity_2IdentityPartitionedCall:output:2*
T0	*'
_output_shapes
:˙˙˙˙˙˙˙˙˙b

Identity_3IdentityPartitionedCall:output:3*
T0	*'
_output_shapes
:˙˙˙˙˙˙˙˙˙b

Identity_4IdentityPartitionedCall:output:4*
T0	*'
_output_shapes
:˙˙˙˙˙˙˙˙˙"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*(
_construction_contextkEagerRuntime*z
_input_shapesi
g:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙: : : : :O K
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:QM
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
"
_user_specified_name
inputs_1:QM
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
"
_user_specified_name
inputs_2:QM
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
"
_user_specified_name
inputs_3:QM
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
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
$
Ą
__inference_pruned_1213

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
:˙˙˙˙˙˙˙˙˙t
scale_to_z_score/CastCastinputs_copy:output:0*

DstT0*

SrcT0	*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
scale_to_z_score/subSubscale_to_z_score/Cast:y:0scale_to_z_score_sub_y*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙t
scale_to_z_score/zeros_like	ZerosLikescale_to_z_score/sub:z:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙W
scale_to_z_score/SqrtSqrtscale_to_z_score_sqrt_x*
T0*
_output_shapes
: 
scale_to_z_score/NotEqualNotEqualscale_to_z_score/Sqrt:y:0$scale_to_z_score/NotEqual/y:output:0*
T0*
_output_shapes
: n
scale_to_z_score/Cast_1Castscale_to_z_score/NotEqual:z:0*

DstT0*

SrcT0
*
_output_shapes
: 
scale_to_z_score/addAddV2scale_to_z_score/zeros_like:y:0scale_to_z_score/Cast_1:y:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙z
scale_to_z_score/Cast_2Castscale_to_z_score/add:z:0*

DstT0
*

SrcT0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
scale_to_z_score/truedivRealDivscale_to_z_score/sub:z:0scale_to_z_score/Sqrt:y:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙Ź
scale_to_z_score/SelectV2SelectV2scale_to_z_score/Cast_2:y:0scale_to_z_score/truediv:z:0scale_to_z_score/sub:z:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙j
IdentityIdentity"scale_to_z_score/SelectV2:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙U
inputs_1_copyIdentityinputs_1*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
scale_to_z_score_1/subSubinputs_1_copy:output:0scale_to_z_score_1_sub_y*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙x
scale_to_z_score_1/zeros_like	ZerosLikescale_to_z_score_1/sub:z:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙[
scale_to_z_score_1/SqrtSqrtscale_to_z_score_1_sqrt_x*
T0*
_output_shapes
: 
scale_to_z_score_1/NotEqualNotEqualscale_to_z_score_1/Sqrt:y:0&scale_to_z_score_1/NotEqual/y:output:0*
T0*
_output_shapes
: p
scale_to_z_score_1/CastCastscale_to_z_score_1/NotEqual:z:0*

DstT0*

SrcT0
*
_output_shapes
: 
scale_to_z_score_1/addAddV2!scale_to_z_score_1/zeros_like:y:0scale_to_z_score_1/Cast:y:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙~
scale_to_z_score_1/Cast_1Castscale_to_z_score_1/add:z:0*

DstT0
*

SrcT0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
scale_to_z_score_1/truedivRealDivscale_to_z_score_1/sub:z:0scale_to_z_score_1/Sqrt:y:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙´
scale_to_z_score_1/SelectV2SelectV2scale_to_z_score_1/Cast_1:y:0scale_to_z_score_1/truediv:z:0scale_to_z_score_1/sub:z:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙n

Identity_1Identity$scale_to_z_score_1/SelectV2:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙U
inputs_2_copyIdentityinputs_2*
T0	*'
_output_shapes
:˙˙˙˙˙˙˙˙˙`

Identity_2Identityinputs_2_copy:output:0*
T0	*'
_output_shapes
:˙˙˙˙˙˙˙˙˙U
inputs_3_copyIdentityinputs_3*
T0	*'
_output_shapes
:˙˙˙˙˙˙˙˙˙`

Identity_3Identityinputs_3_copy:output:0*
T0	*'
_output_shapes
:˙˙˙˙˙˙˙˙˙U
inputs_4_copyIdentityinputs_4*
T0	*'
_output_shapes
:˙˙˙˙˙˙˙˙˙`

Identity_4Identityinputs_4_copy:output:0*
T0	*'
_output_shapes
:˙˙˙˙˙˙˙˙˙"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*(
_construction_contextkEagerRuntime*z
_input_shapesi
g:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙: : : : :- )
'
_output_shapes
:˙˙˙˙˙˙˙˙˙:-)
'
_output_shapes
:˙˙˙˙˙˙˙˙˙:-)
'
_output_shapes
:˙˙˙˙˙˙˙˙˙:-)
'
_output_shapes
:˙˙˙˙˙˙˙˙˙:-)
'
_output_shapes
:˙˙˙˙˙˙˙˙˙:
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

l
__inference__traced_save_1273
file_prefix
savev2_const_4

identity_1˘MergeV2Checkpointsw
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
_temp/part
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPHo
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B Ú
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0savev2_const_4"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtypes
2
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:ł
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: 7
NoOpNoOp^MergeV2Checkpoints*
_output_shapes
 "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:?;

_output_shapes
: 
!
_user_specified_name	Const_4"ćJ
saver_filename:0StatefulPartitionedCall:0StatefulPartitionedCall_18"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*
serving_defaultř
9
inputs/
serving_default_inputs:0	˙˙˙˙˙˙˙˙˙
=
inputs_11
serving_default_inputs_1:0˙˙˙˙˙˙˙˙˙
=
inputs_21
serving_default_inputs_2:0	˙˙˙˙˙˙˙˙˙
=
inputs_31
serving_default_inputs_3:0	˙˙˙˙˙˙˙˙˙
=
inputs_41
serving_default_inputs_4:0	˙˙˙˙˙˙˙˙˙2
age_xf(
PartitionedCall:0˙˙˙˙˙˙˙˙˙2
bmi_xf(
PartitionedCall:1˙˙˙˙˙˙˙˙˙5
	gender_xf(
PartitionedCall:2	˙˙˙˙˙˙˙˙˙>
obesitycategory_xf(
PartitionedCall:3	˙˙˙˙˙˙˙˙˙D
physicalactivitylevel_xf(
PartitionedCall:4	˙˙˙˙˙˙˙˙˙tensorflow/serving/predict:Ŕ

created_variables
	resources
trackable_objects
initializers

assets
transform_fn

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
Ĺ
	capture_0
		capture_1

	capture_2
	capture_3BK
__inference_pruned_1213inputsinputs_1inputs_2inputs_3inputs_4z	capture_0z		capture_1z
	capture_2z	capture_3
,
serving_default"
signature_map
!J	
Const_3jtf.TrackableConstant
!J	
Const_2jtf.TrackableConstant
!J	
Const_1jtf.TrackableConstant
J
Constjtf.TrackableConstant

	capture_0
		capture_1

	capture_2
	capture_3B
"__inference_signature_wrapper_1239inputsinputs_1inputs_2inputs_3inputs_4"Č
Á˛˝
FullArgSpec
args 
varargs
 
varkw
 
defaults
 K

kwonlyargs=:
jinputs

jinputs_1

jinputs_2

jinputs_3

jinputs_4
kwonlydefaults
 
annotationsŞ *
 z	capture_0z		capture_1z
	capture_2z	capture_3
__inference_pruned_1213č	
ş˘ś
Ž˘Ş
§ŞŁ
+
Age$!

inputs_age˙˙˙˙˙˙˙˙˙	
+
BMI$!

inputs_bmi˙˙˙˙˙˙˙˙˙
1
Gender'$
inputs_gender˙˙˙˙˙˙˙˙˙	
C
ObesityCategory0-
inputs_obesitycategory˙˙˙˙˙˙˙˙˙	
O
PhysicalActivityLevel63
inputs_physicalactivitylevel˙˙˙˙˙˙˙˙˙	
Ş "˘Ş
*
age_xf 
age_xf˙˙˙˙˙˙˙˙˙
*
bmi_xf 
bmi_xf˙˙˙˙˙˙˙˙˙
0
	gender_xf# 
	gender_xf˙˙˙˙˙˙˙˙˙	
B
obesitycategory_xf,)
obesitycategory_xf˙˙˙˙˙˙˙˙˙	
N
physicalactivitylevel_xf2/
physicalactivitylevel_xf˙˙˙˙˙˙˙˙˙	Ń
"__inference_signature_wrapper_1239Ş	
ü˘ř
˘ 
đŞě
*
inputs 
inputs˙˙˙˙˙˙˙˙˙	
.
inputs_1"
inputs_1˙˙˙˙˙˙˙˙˙
.
inputs_2"
inputs_2˙˙˙˙˙˙˙˙˙	
.
inputs_3"
inputs_3˙˙˙˙˙˙˙˙˙	
.
inputs_4"
inputs_4˙˙˙˙˙˙˙˙˙	"˘Ş
*
age_xf 
age_xf˙˙˙˙˙˙˙˙˙
*
bmi_xf 
bmi_xf˙˙˙˙˙˙˙˙˙
0
	gender_xf# 
	gender_xf˙˙˙˙˙˙˙˙˙	
B
obesitycategory_xf,)
obesitycategory_xf˙˙˙˙˙˙˙˙˙	
N
physicalactivitylevel_xf2/
physicalactivitylevel_xf˙˙˙˙˙˙˙˙˙	