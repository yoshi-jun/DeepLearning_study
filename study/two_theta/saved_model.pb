��
��
B
AddV2
x"T
y"T
z"T"
Ttype:
2	��
B
AssignVariableOp
resource
value"dtype"
dtypetype�
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
�
Conv3D

input"T
filter"T
output"T"
Ttype:
2"
strides	list(int)(0""
paddingstring:
SAMEVALID"0
data_formatstringNDHWC:
NDHWCNCDHW"!
	dilations	list(int)	

.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(�
=
Mul
x"T
y"T
z"T"
Ttype:
2	�

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype�
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
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
executor_typestring �
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
;
Sub
x"T
y"T
z"T"
Ttype:
2	
�
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �"serve*2.4.12unknown8ա
�
conv3d_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameconv3d_1/kernel

#conv3d_1/kernel/Read/ReadVariableOpReadVariableOpconv3d_1/kernel**
_output_shapes
:*
dtype0
r
conv3d_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv3d_1/bias
k
!conv3d_1/bias/Read/ReadVariableOpReadVariableOpconv3d_1/bias*
_output_shapes
:*
dtype0
z
normalize_1/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_namenormalize_1/gamma
s
%normalize_1/gamma/Read/ReadVariableOpReadVariableOpnormalize_1/gamma*
_output_shapes
:*
dtype0
x
normalize_1/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_namenormalize_1/beta
q
$normalize_1/beta/Read/ReadVariableOpReadVariableOpnormalize_1/beta*
_output_shapes
:*
dtype0
�
normalize_1/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_namenormalize_1/moving_mean

+normalize_1/moving_mean/Read/ReadVariableOpReadVariableOpnormalize_1/moving_mean*
_output_shapes
:*
dtype0
�
normalize_1/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namenormalize_1/moving_variance
�
/normalize_1/moving_variance/Read/ReadVariableOpReadVariableOpnormalize_1/moving_variance*
_output_shapes
:*
dtype0
�
conv3d_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: * 
shared_nameconv3d_2/kernel

#conv3d_2/kernel/Read/ReadVariableOpReadVariableOpconv3d_2/kernel**
_output_shapes
: *
dtype0
r
conv3d_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv3d_2/bias
k
!conv3d_2/bias/Read/ReadVariableOpReadVariableOpconv3d_2/bias*
_output_shapes
: *
dtype0
z
normalize_2/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *"
shared_namenormalize_2/gamma
s
%normalize_2/gamma/Read/ReadVariableOpReadVariableOpnormalize_2/gamma*
_output_shapes
: *
dtype0
x
normalize_2/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *!
shared_namenormalize_2/beta
q
$normalize_2/beta/Read/ReadVariableOpReadVariableOpnormalize_2/beta*
_output_shapes
: *
dtype0
�
normalize_2/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape: *(
shared_namenormalize_2/moving_mean

+normalize_2/moving_mean/Read/ReadVariableOpReadVariableOpnormalize_2/moving_mean*
_output_shapes
: *
dtype0
�
normalize_2/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape: *,
shared_namenormalize_2/moving_variance
�
/normalize_2/moving_variance/Read/ReadVariableOpReadVariableOpnormalize_2/moving_variance*
_output_shapes
: *
dtype0
�
conv3d_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: @* 
shared_nameconv3d_3/kernel

#conv3d_3/kernel/Read/ReadVariableOpReadVariableOpconv3d_3/kernel**
_output_shapes
: @*
dtype0
r
conv3d_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv3d_3/bias
k
!conv3d_3/bias/Read/ReadVariableOpReadVariableOpconv3d_3/bias*
_output_shapes
:@*
dtype0
w
theta/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:���*
shared_nametheta/kernel
p
 theta/kernel/Read/ReadVariableOpReadVariableOptheta/kernel*!
_output_shapes
:���*
dtype0
m

theta/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_name
theta/bias
f
theta/bias/Read/ReadVariableOpReadVariableOp
theta/bias*
_output_shapes	
:�*
dtype0
d
SGD/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name
SGD/iter
]
SGD/iter/Read/ReadVariableOpReadVariableOpSGD/iter*
_output_shapes
: *
dtype0	
f
	SGD/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	SGD/decay
_
SGD/decay/Read/ReadVariableOpReadVariableOp	SGD/decay*
_output_shapes
: *
dtype0
v
SGD/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *"
shared_nameSGD/learning_rate
o
%SGD/learning_rate/Read/ReadVariableOpReadVariableOpSGD/learning_rate*
_output_shapes
: *
dtype0
l
SGD/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameSGD/momentum
e
 SGD/momentum/Read/ReadVariableOpReadVariableOpSGD/momentum*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0

NoOpNoOp
�*
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�*
value�*B�* B�*
�
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
layer_with_weights-4
layer-5
layer-6
layer_with_weights-5
layer-7
		optimizer

regularization_losses
trainable_variables
	variables
	keras_api

signatures
 
h

kernel
bias
regularization_losses
trainable_variables
	variables
	keras_api
�
axis
	gamma
beta
moving_mean
moving_variance
regularization_losses
trainable_variables
	variables
	keras_api
h

kernel
bias
 regularization_losses
!trainable_variables
"	variables
#	keras_api
�
$axis
	%gamma
&beta
'moving_mean
(moving_variance
)regularization_losses
*trainable_variables
+	variables
,	keras_api
h

-kernel
.bias
/regularization_losses
0trainable_variables
1	variables
2	keras_api
R
3regularization_losses
4trainable_variables
5	variables
6	keras_api
h

7kernel
8bias
9regularization_losses
:trainable_variables
;	variables
<	keras_api
6
=iter
	>decay
?learning_rate
@momentum
 
V
0
1
2
3
4
5
%6
&7
-8
.9
710
811
v
0
1
2
3
4
5
6
7
%8
&9
'10
(11
-12
.13
714
815
�
Alayer_metrics

Blayers
Cnon_trainable_variables
Dmetrics

regularization_losses
trainable_variables
Elayer_regularization_losses
	variables
 
[Y
VARIABLE_VALUEconv3d_1/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv3d_1/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
�
Flayer_metrics

Glayers
Hmetrics
Inon_trainable_variables
regularization_losses
trainable_variables
Jlayer_regularization_losses
	variables
 
\Z
VARIABLE_VALUEnormalize_1/gamma5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEnormalize_1/beta4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUE
hf
VARIABLE_VALUEnormalize_1/moving_mean;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
pn
VARIABLE_VALUEnormalize_1/moving_variance?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
2
3
�
Klayer_metrics

Llayers
Mmetrics
Nnon_trainable_variables
regularization_losses
trainable_variables
Olayer_regularization_losses
	variables
[Y
VARIABLE_VALUEconv3d_2/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv3d_2/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
�
Player_metrics

Qlayers
Rmetrics
Snon_trainable_variables
 regularization_losses
!trainable_variables
Tlayer_regularization_losses
"	variables
 
\Z
VARIABLE_VALUEnormalize_2/gamma5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEnormalize_2/beta4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUE
hf
VARIABLE_VALUEnormalize_2/moving_mean;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
pn
VARIABLE_VALUEnormalize_2/moving_variance?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 

%0
&1

%0
&1
'2
(3
�
Ulayer_metrics

Vlayers
Wmetrics
Xnon_trainable_variables
)regularization_losses
*trainable_variables
Ylayer_regularization_losses
+	variables
[Y
VARIABLE_VALUEconv3d_3/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv3d_3/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE
 

-0
.1

-0
.1
�
Zlayer_metrics

[layers
\metrics
]non_trainable_variables
/regularization_losses
0trainable_variables
^layer_regularization_losses
1	variables
 
 
 
�
_layer_metrics

`layers
ametrics
bnon_trainable_variables
3regularization_losses
4trainable_variables
clayer_regularization_losses
5	variables
XV
VARIABLE_VALUEtheta/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUE
theta/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE
 

70
81

70
81
�
dlayer_metrics

elayers
fmetrics
gnon_trainable_variables
9regularization_losses
:trainable_variables
hlayer_regularization_losses
;	variables
GE
VARIABLE_VALUESGD/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
IG
VARIABLE_VALUE	SGD/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUESGD/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUESGD/momentum-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
 
8
0
1
2
3
4
5
6
7

0
1
'2
(3

i0
 
 
 
 
 
 
 
 
 

0
1
 
 
 
 
 
 
 
 
 

'0
(1
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
4
	jtotal
	kcount
l	variables
m	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

j0
k1

l	variables
�
serving_default_input_7Placeholder*4
_output_shapes"
 :���������==�*
dtype0*)
shape :���������==�
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_7conv3d_1/kernelconv3d_1/biasnormalize_1/moving_variancenormalize_1/gammanormalize_1/moving_meannormalize_1/betaconv3d_2/kernelconv3d_2/biasnormalize_2/moving_variancenormalize_2/gammanormalize_2/moving_meannormalize_2/betaconv3d_3/kernelconv3d_3/biastheta/kernel
theta/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*2
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*1J 8� *,
f'R%
#__inference_signature_wrapper_35487
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename#conv3d_1/kernel/Read/ReadVariableOp!conv3d_1/bias/Read/ReadVariableOp%normalize_1/gamma/Read/ReadVariableOp$normalize_1/beta/Read/ReadVariableOp+normalize_1/moving_mean/Read/ReadVariableOp/normalize_1/moving_variance/Read/ReadVariableOp#conv3d_2/kernel/Read/ReadVariableOp!conv3d_2/bias/Read/ReadVariableOp%normalize_2/gamma/Read/ReadVariableOp$normalize_2/beta/Read/ReadVariableOp+normalize_2/moving_mean/Read/ReadVariableOp/normalize_2/moving_variance/Read/ReadVariableOp#conv3d_3/kernel/Read/ReadVariableOp!conv3d_3/bias/Read/ReadVariableOp theta/kernel/Read/ReadVariableOptheta/bias/Read/ReadVariableOpSGD/iter/Read/ReadVariableOpSGD/decay/Read/ReadVariableOp%SGD/learning_rate/Read/ReadVariableOp SGD/momentum/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOpConst*#
Tin
2	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*1J 8� *'
f"R 
__inference__traced_save_36233
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv3d_1/kernelconv3d_1/biasnormalize_1/gammanormalize_1/betanormalize_1/moving_meannormalize_1/moving_varianceconv3d_2/kernelconv3d_2/biasnormalize_2/gammanormalize_2/betanormalize_2/moving_meannormalize_2/moving_varianceconv3d_3/kernelconv3d_3/biastheta/kernel
theta/biasSGD/iter	SGD/decaySGD/learning_rateSGD/momentumtotalcount*"
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*1J 8� **
f%R#
!__inference__traced_restore_36309��
�	
�
@__inference_theta_layer_call_and_return_conditional_losses_36135

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*!
_output_shapes
:���*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2	
BiasAddb
SoftmaxSoftmaxBiasAdd:output:0*
T0*(
_output_shapes
:����������2	
Softmax�
IdentityIdentitySoftmax:softmax:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*0
_input_shapes
:�����������::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:Q M
)
_output_shapes
:�����������
 
_user_specified_nameinputs
�
�
F__inference_normalize_2_layer_call_and_return_conditional_losses_34918

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identity��batchnorm/ReadVariableOp�batchnorm/ReadVariableOp_1�batchnorm/ReadVariableOp_2�batchnorm/mul/ReadVariableOp�
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:2
batchnorm/add/y�
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
: 2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
: 2
batchnorm/Rsqrt�
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/mul/ReadVariableOp�
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 2
batchnorm/mul�
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*N
_output_shapes<
::8������������������������������������ 2
batchnorm/mul_1�
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOp_1�
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
: 2
batchnorm/mul_2�
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOp_2�
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
: 2
batchnorm/sub�
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*N
_output_shapes<
::8������������������������������������ 2
batchnorm/add_1�
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*N
_output_shapes<
::8������������������������������������ 2

Identity"
identityIdentity:output:0*]
_input_shapesL
J:8������������������������������������ ::::24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:v r
N
_output_shapes<
::8������������������������������������ 
 
_user_specified_nameinputs
�

�
C__inference_conv3d_2_layer_call_and_return_conditional_losses_35062

inputs"
conv3d_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�Conv3D/ReadVariableOp�
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
: *
dtype02
Conv3D/ReadVariableOp�
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������% *
paddingVALID*
strides	
2
Conv3D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������% 2	
BiasAddd
ReluReluBiasAdd:output:0*
T0*3
_output_shapes!
:���������% 2
Relu�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*
T0*3
_output_shapes!
:���������% 2

Identity"
identityIdentity:output:0*:
_input_shapes)
':���������K::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:[ W
3
_output_shapes!
:���������K
 
_user_specified_nameinputs
�
�
F__inference_normalize_1_layer_call_and_return_conditional_losses_35015

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identity��batchnorm/ReadVariableOp�batchnorm/ReadVariableOp_1�batchnorm/ReadVariableOp_2�batchnorm/mul/ReadVariableOp�
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:2
batchnorm/add/y�
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/Rsqrt�
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/mul/ReadVariableOp�
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mul�
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*3
_output_shapes!
:���������K2
batchnorm/mul_1�
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp_1�
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2�
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp_2�
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/sub�
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*3
_output_shapes!
:���������K2
batchnorm/add_1�
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*3
_output_shapes!
:���������K2

Identity"
identityIdentity:output:0*B
_input_shapes1
/:���������K::::24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:[ W
3
_output_shapes!
:���������K
 
_user_specified_nameinputs
�Z
�
B__inference_model_6_layer_call_and_return_conditional_losses_35651

inputs+
'conv3d_1_conv3d_readvariableop_resource,
(conv3d_1_biasadd_readvariableop_resource1
-normalize_1_batchnorm_readvariableop_resource5
1normalize_1_batchnorm_mul_readvariableop_resource3
/normalize_1_batchnorm_readvariableop_1_resource3
/normalize_1_batchnorm_readvariableop_2_resource+
'conv3d_2_conv3d_readvariableop_resource,
(conv3d_2_biasadd_readvariableop_resource1
-normalize_2_batchnorm_readvariableop_resource5
1normalize_2_batchnorm_mul_readvariableop_resource3
/normalize_2_batchnorm_readvariableop_1_resource3
/normalize_2_batchnorm_readvariableop_2_resource+
'conv3d_3_conv3d_readvariableop_resource,
(conv3d_3_biasadd_readvariableop_resource(
$theta_matmul_readvariableop_resource)
%theta_biasadd_readvariableop_resource
identity��conv3d_1/BiasAdd/ReadVariableOp�conv3d_1/Conv3D/ReadVariableOp�conv3d_2/BiasAdd/ReadVariableOp�conv3d_2/Conv3D/ReadVariableOp�conv3d_3/BiasAdd/ReadVariableOp�conv3d_3/Conv3D/ReadVariableOp�$normalize_1/batchnorm/ReadVariableOp�&normalize_1/batchnorm/ReadVariableOp_1�&normalize_1/batchnorm/ReadVariableOp_2�(normalize_1/batchnorm/mul/ReadVariableOp�$normalize_2/batchnorm/ReadVariableOp�&normalize_2/batchnorm/ReadVariableOp_1�&normalize_2/batchnorm/ReadVariableOp_2�(normalize_2/batchnorm/mul/ReadVariableOp�theta/BiasAdd/ReadVariableOp�theta/MatMul/ReadVariableOp�
conv3d_1/Conv3D/ReadVariableOpReadVariableOp'conv3d_1_conv3d_readvariableop_resource**
_output_shapes
:*
dtype02 
conv3d_1/Conv3D/ReadVariableOp�
conv3d_1/Conv3DConv3Dinputs&conv3d_1/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������K*
paddingVALID*
strides	
2
conv3d_1/Conv3D�
conv3d_1/BiasAdd/ReadVariableOpReadVariableOp(conv3d_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
conv3d_1/BiasAdd/ReadVariableOp�
conv3d_1/BiasAddBiasAddconv3d_1/Conv3D:output:0'conv3d_1/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������K2
conv3d_1/BiasAdd
conv3d_1/ReluReluconv3d_1/BiasAdd:output:0*
T0*3
_output_shapes!
:���������K2
conv3d_1/Relu�
$normalize_1/batchnorm/ReadVariableOpReadVariableOp-normalize_1_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02&
$normalize_1/batchnorm/ReadVariableOp
normalize_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:2
normalize_1/batchnorm/add/y�
normalize_1/batchnorm/addAddV2,normalize_1/batchnorm/ReadVariableOp:value:0$normalize_1/batchnorm/add/y:output:0*
T0*
_output_shapes
:2
normalize_1/batchnorm/add�
normalize_1/batchnorm/RsqrtRsqrtnormalize_1/batchnorm/add:z:0*
T0*
_output_shapes
:2
normalize_1/batchnorm/Rsqrt�
(normalize_1/batchnorm/mul/ReadVariableOpReadVariableOp1normalize_1_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02*
(normalize_1/batchnorm/mul/ReadVariableOp�
normalize_1/batchnorm/mulMulnormalize_1/batchnorm/Rsqrt:y:00normalize_1/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2
normalize_1/batchnorm/mul�
normalize_1/batchnorm/mul_1Mulconv3d_1/Relu:activations:0normalize_1/batchnorm/mul:z:0*
T0*3
_output_shapes!
:���������K2
normalize_1/batchnorm/mul_1�
&normalize_1/batchnorm/ReadVariableOp_1ReadVariableOp/normalize_1_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype02(
&normalize_1/batchnorm/ReadVariableOp_1�
normalize_1/batchnorm/mul_2Mul.normalize_1/batchnorm/ReadVariableOp_1:value:0normalize_1/batchnorm/mul:z:0*
T0*
_output_shapes
:2
normalize_1/batchnorm/mul_2�
&normalize_1/batchnorm/ReadVariableOp_2ReadVariableOp/normalize_1_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype02(
&normalize_1/batchnorm/ReadVariableOp_2�
normalize_1/batchnorm/subSub.normalize_1/batchnorm/ReadVariableOp_2:value:0normalize_1/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
normalize_1/batchnorm/sub�
normalize_1/batchnorm/add_1AddV2normalize_1/batchnorm/mul_1:z:0normalize_1/batchnorm/sub:z:0*
T0*3
_output_shapes!
:���������K2
normalize_1/batchnorm/add_1�
conv3d_2/Conv3D/ReadVariableOpReadVariableOp'conv3d_2_conv3d_readvariableop_resource**
_output_shapes
: *
dtype02 
conv3d_2/Conv3D/ReadVariableOp�
conv3d_2/Conv3DConv3Dnormalize_1/batchnorm/add_1:z:0&conv3d_2/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������% *
paddingVALID*
strides	
2
conv3d_2/Conv3D�
conv3d_2/BiasAdd/ReadVariableOpReadVariableOp(conv3d_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
conv3d_2/BiasAdd/ReadVariableOp�
conv3d_2/BiasAddBiasAddconv3d_2/Conv3D:output:0'conv3d_2/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������% 2
conv3d_2/BiasAdd
conv3d_2/ReluReluconv3d_2/BiasAdd:output:0*
T0*3
_output_shapes!
:���������% 2
conv3d_2/Relu�
$normalize_2/batchnorm/ReadVariableOpReadVariableOp-normalize_2_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype02&
$normalize_2/batchnorm/ReadVariableOp
normalize_2/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:2
normalize_2/batchnorm/add/y�
normalize_2/batchnorm/addAddV2,normalize_2/batchnorm/ReadVariableOp:value:0$normalize_2/batchnorm/add/y:output:0*
T0*
_output_shapes
: 2
normalize_2/batchnorm/add�
normalize_2/batchnorm/RsqrtRsqrtnormalize_2/batchnorm/add:z:0*
T0*
_output_shapes
: 2
normalize_2/batchnorm/Rsqrt�
(normalize_2/batchnorm/mul/ReadVariableOpReadVariableOp1normalize_2_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02*
(normalize_2/batchnorm/mul/ReadVariableOp�
normalize_2/batchnorm/mulMulnormalize_2/batchnorm/Rsqrt:y:00normalize_2/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 2
normalize_2/batchnorm/mul�
normalize_2/batchnorm/mul_1Mulconv3d_2/Relu:activations:0normalize_2/batchnorm/mul:z:0*
T0*3
_output_shapes!
:���������% 2
normalize_2/batchnorm/mul_1�
&normalize_2/batchnorm/ReadVariableOp_1ReadVariableOp/normalize_2_batchnorm_readvariableop_1_resource*
_output_shapes
: *
dtype02(
&normalize_2/batchnorm/ReadVariableOp_1�
normalize_2/batchnorm/mul_2Mul.normalize_2/batchnorm/ReadVariableOp_1:value:0normalize_2/batchnorm/mul:z:0*
T0*
_output_shapes
: 2
normalize_2/batchnorm/mul_2�
&normalize_2/batchnorm/ReadVariableOp_2ReadVariableOp/normalize_2_batchnorm_readvariableop_2_resource*
_output_shapes
: *
dtype02(
&normalize_2/batchnorm/ReadVariableOp_2�
normalize_2/batchnorm/subSub.normalize_2/batchnorm/ReadVariableOp_2:value:0normalize_2/batchnorm/mul_2:z:0*
T0*
_output_shapes
: 2
normalize_2/batchnorm/sub�
normalize_2/batchnorm/add_1AddV2normalize_2/batchnorm/mul_1:z:0normalize_2/batchnorm/sub:z:0*
T0*3
_output_shapes!
:���������% 2
normalize_2/batchnorm/add_1�
conv3d_3/Conv3D/ReadVariableOpReadVariableOp'conv3d_3_conv3d_readvariableop_resource**
_output_shapes
: @*
dtype02 
conv3d_3/Conv3D/ReadVariableOp�
conv3d_3/Conv3DConv3Dnormalize_2/batchnorm/add_1:z:0&conv3d_3/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������@*
paddingVALID*
strides	
2
conv3d_3/Conv3D�
conv3d_3/BiasAdd/ReadVariableOpReadVariableOp(conv3d_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv3d_3/BiasAdd/ReadVariableOp�
conv3d_3/BiasAddBiasAddconv3d_3/Conv3D:output:0'conv3d_3/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������@2
conv3d_3/BiasAdd
conv3d_3/ReluReluconv3d_3/BiasAdd:output:0*
T0*3
_output_shapes!
:���������@2
conv3d_3/Reluo
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"������  2
flatten/Const�
flatten/ReshapeReshapeconv3d_3/Relu:activations:0flatten/Const:output:0*
T0*)
_output_shapes
:�����������2
flatten/Reshape�
theta/MatMul/ReadVariableOpReadVariableOp$theta_matmul_readvariableop_resource*!
_output_shapes
:���*
dtype02
theta/MatMul/ReadVariableOp�
theta/MatMulMatMulflatten/Reshape:output:0#theta/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
theta/MatMul�
theta/BiasAdd/ReadVariableOpReadVariableOp%theta_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
theta/BiasAdd/ReadVariableOp�
theta/BiasAddBiasAddtheta/MatMul:product:0$theta/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
theta/BiasAddt
theta/SoftmaxSoftmaxtheta/BiasAdd:output:0*
T0*(
_output_shapes
:����������2
theta/Softmax�
IdentityIdentitytheta/Softmax:softmax:0 ^conv3d_1/BiasAdd/ReadVariableOp^conv3d_1/Conv3D/ReadVariableOp ^conv3d_2/BiasAdd/ReadVariableOp^conv3d_2/Conv3D/ReadVariableOp ^conv3d_3/BiasAdd/ReadVariableOp^conv3d_3/Conv3D/ReadVariableOp%^normalize_1/batchnorm/ReadVariableOp'^normalize_1/batchnorm/ReadVariableOp_1'^normalize_1/batchnorm/ReadVariableOp_2)^normalize_1/batchnorm/mul/ReadVariableOp%^normalize_2/batchnorm/ReadVariableOp'^normalize_2/batchnorm/ReadVariableOp_1'^normalize_2/batchnorm/ReadVariableOp_2)^normalize_2/batchnorm/mul/ReadVariableOp^theta/BiasAdd/ReadVariableOp^theta/MatMul/ReadVariableOp*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*s
_input_shapesb
`:���������==�::::::::::::::::2B
conv3d_1/BiasAdd/ReadVariableOpconv3d_1/BiasAdd/ReadVariableOp2@
conv3d_1/Conv3D/ReadVariableOpconv3d_1/Conv3D/ReadVariableOp2B
conv3d_2/BiasAdd/ReadVariableOpconv3d_2/BiasAdd/ReadVariableOp2@
conv3d_2/Conv3D/ReadVariableOpconv3d_2/Conv3D/ReadVariableOp2B
conv3d_3/BiasAdd/ReadVariableOpconv3d_3/BiasAdd/ReadVariableOp2@
conv3d_3/Conv3D/ReadVariableOpconv3d_3/Conv3D/ReadVariableOp2L
$normalize_1/batchnorm/ReadVariableOp$normalize_1/batchnorm/ReadVariableOp2P
&normalize_1/batchnorm/ReadVariableOp_1&normalize_1/batchnorm/ReadVariableOp_12P
&normalize_1/batchnorm/ReadVariableOp_2&normalize_1/batchnorm/ReadVariableOp_22T
(normalize_1/batchnorm/mul/ReadVariableOp(normalize_1/batchnorm/mul/ReadVariableOp2L
$normalize_2/batchnorm/ReadVariableOp$normalize_2/batchnorm/ReadVariableOp2P
&normalize_2/batchnorm/ReadVariableOp_1&normalize_2/batchnorm/ReadVariableOp_12P
&normalize_2/batchnorm/ReadVariableOp_2&normalize_2/batchnorm/ReadVariableOp_22T
(normalize_2/batchnorm/mul/ReadVariableOp(normalize_2/batchnorm/mul/ReadVariableOp2<
theta/BiasAdd/ReadVariableOptheta/BiasAdd/ReadVariableOp2:
theta/MatMul/ReadVariableOptheta/MatMul/ReadVariableOp:\ X
4
_output_shapes"
 :���������==�
 
_user_specified_nameinputs
�%
�
B__inference_model_6_layer_call_and_return_conditional_losses_35407

inputs
conv3d_1_35367
conv3d_1_35369
normalize_1_35372
normalize_1_35374
normalize_1_35376
normalize_1_35378
conv3d_2_35381
conv3d_2_35383
normalize_2_35386
normalize_2_35388
normalize_2_35390
normalize_2_35392
conv3d_3_35395
conv3d_3_35397
theta_35401
theta_35403
identity�� conv3d_1/StatefulPartitionedCall� conv3d_2/StatefulPartitionedCall� conv3d_3/StatefulPartitionedCall�#normalize_1/StatefulPartitionedCall�#normalize_2/StatefulPartitionedCall�theta/StatefulPartitionedCall�
 conv3d_1/StatefulPartitionedCallStatefulPartitionedCallinputsconv3d_1_35367conv3d_1_35369*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������K*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*1J 8� *L
fGRE
C__inference_conv3d_1_layer_call_and_return_conditional_losses_349442"
 conv3d_1/StatefulPartitionedCall�
#normalize_1/StatefulPartitionedCallStatefulPartitionedCall)conv3d_1/StatefulPartitionedCall:output:0normalize_1_35372normalize_1_35374normalize_1_35376normalize_1_35378*
Tin	
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������K*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*1J 8� *O
fJRH
F__inference_normalize_1_layer_call_and_return_conditional_losses_350152%
#normalize_1/StatefulPartitionedCall�
 conv3d_2/StatefulPartitionedCallStatefulPartitionedCall,normalize_1/StatefulPartitionedCall:output:0conv3d_2_35381conv3d_2_35383*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������% *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*1J 8� *L
fGRE
C__inference_conv3d_2_layer_call_and_return_conditional_losses_350622"
 conv3d_2/StatefulPartitionedCall�
#normalize_2/StatefulPartitionedCallStatefulPartitionedCall)conv3d_2/StatefulPartitionedCall:output:0normalize_2_35386normalize_2_35388normalize_2_35390normalize_2_35392*
Tin	
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������% *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*1J 8� *O
fJRH
F__inference_normalize_2_layer_call_and_return_conditional_losses_351332%
#normalize_2/StatefulPartitionedCall�
 conv3d_3/StatefulPartitionedCallStatefulPartitionedCall,normalize_2/StatefulPartitionedCall:output:0conv3d_3_35395conv3d_3_35397*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*1J 8� *L
fGRE
C__inference_conv3d_3_layer_call_and_return_conditional_losses_351802"
 conv3d_3/StatefulPartitionedCall�
flatten/PartitionedCallPartitionedCall)conv3d_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:�����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*1J 8� *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_352022
flatten/PartitionedCall�
theta/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0theta_35401theta_35403*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*1J 8� *I
fDRB
@__inference_theta_layer_call_and_return_conditional_losses_352212
theta/StatefulPartitionedCall�
IdentityIdentity&theta/StatefulPartitionedCall:output:0!^conv3d_1/StatefulPartitionedCall!^conv3d_2/StatefulPartitionedCall!^conv3d_3/StatefulPartitionedCall$^normalize_1/StatefulPartitionedCall$^normalize_2/StatefulPartitionedCall^theta/StatefulPartitionedCall*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*s
_input_shapesb
`:���������==�::::::::::::::::2D
 conv3d_1/StatefulPartitionedCall conv3d_1/StatefulPartitionedCall2D
 conv3d_2/StatefulPartitionedCall conv3d_2/StatefulPartitionedCall2D
 conv3d_3/StatefulPartitionedCall conv3d_3/StatefulPartitionedCall2J
#normalize_1/StatefulPartitionedCall#normalize_1/StatefulPartitionedCall2J
#normalize_2/StatefulPartitionedCall#normalize_2/StatefulPartitionedCall2>
theta/StatefulPartitionedCalltheta/StatefulPartitionedCall:\ X
4
_output_shapes"
 :���������==�
 
_user_specified_nameinputs
�2
�
F__inference_normalize_2_layer_call_and_return_conditional_losses_35965

inputs
assignmovingavg_35940
assignmovingavg_1_35946)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identity��#AssignMovingAvg/AssignSubVariableOp�AssignMovingAvg/ReadVariableOp�%AssignMovingAvg_1/AssignSubVariableOp� AssignMovingAvg_1/ReadVariableOp�batchnorm/ReadVariableOp�batchnorm/mul/ReadVariableOp�
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*%
valueB"             2 
moments/mean/reduction_indices�
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0**
_output_shapes
: *
	keep_dims(2
moments/mean�
moments/StopGradientStopGradientmoments/mean:output:0*
T0**
_output_shapes
: 2
moments/StopGradient�
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*N
_output_shapes<
::8������������������������������������ 2
moments/SquaredDifference�
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"moments/variance/reduction_indices�
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0**
_output_shapes
: *
	keep_dims(2
moments/variance�
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
: *
squeeze_dims
 2
moments/Squeeze�
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
: *
squeeze_dims
 2
moments/Squeeze_1�
AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:GPU:0*(
_class
loc:@AssignMovingAvg/35940*
_output_shapes
: *
dtype0*
valueB
 *
�#<2
AssignMovingAvg/decay�
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_35940*
_output_shapes
: *
dtype02 
AssignMovingAvg/ReadVariableOp�
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*(
_class
loc:@AssignMovingAvg/35940*
_output_shapes
: 2
AssignMovingAvg/sub�
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*(
_class
loc:@AssignMovingAvg/35940*
_output_shapes
: 2
AssignMovingAvg/mul�
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_35940AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*(
_class
loc:@AssignMovingAvg/35940*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp�
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:GPU:0**
_class 
loc:@AssignMovingAvg_1/35946*
_output_shapes
: *
dtype0*
valueB
 *
�#<2
AssignMovingAvg_1/decay�
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_35946*
_output_shapes
: *
dtype02"
 AssignMovingAvg_1/ReadVariableOp�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0**
_class 
loc:@AssignMovingAvg_1/35946*
_output_shapes
: 2
AssignMovingAvg_1/sub�
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0**
_class 
loc:@AssignMovingAvg_1/35946*
_output_shapes
: 2
AssignMovingAvg_1/mul�
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_35946AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0**
_class 
loc:@AssignMovingAvg_1/35946*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:2
batchnorm/add/y�
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
: 2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
: 2
batchnorm/Rsqrt�
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/mul/ReadVariableOp�
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 2
batchnorm/mul�
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*N
_output_shapes<
::8������������������������������������ 2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
: 2
batchnorm/mul_2�
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOp�
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
: 2
batchnorm/sub�
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*N
_output_shapes<
::8������������������������������������ 2
batchnorm/add_1�
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*N
_output_shapes<
::8������������������������������������ 2

Identity"
identityIdentity:output:0*]
_input_shapesL
J:8������������������������������������ ::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:v r
N
_output_shapes<
::8������������������������������������ 
 
_user_specified_nameinputs
�1
�
F__inference_normalize_1_layer_call_and_return_conditional_losses_34995

inputs
assignmovingavg_34970
assignmovingavg_1_34976)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identity��#AssignMovingAvg/AssignSubVariableOp�AssignMovingAvg/ReadVariableOp�%AssignMovingAvg_1/AssignSubVariableOp� AssignMovingAvg_1/ReadVariableOp�batchnorm/ReadVariableOp�batchnorm/mul/ReadVariableOp�
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*%
valueB"             2 
moments/mean/reduction_indices�
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0**
_output_shapes
:*
	keep_dims(2
moments/mean�
moments/StopGradientStopGradientmoments/mean:output:0*
T0**
_output_shapes
:2
moments/StopGradient�
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*3
_output_shapes!
:���������K2
moments/SquaredDifference�
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"moments/variance/reduction_indices�
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0**
_output_shapes
:*
	keep_dims(2
moments/variance�
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze�
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze_1�
AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:GPU:0*(
_class
loc:@AssignMovingAvg/34970*
_output_shapes
: *
dtype0*
valueB
 *
�#<2
AssignMovingAvg/decay�
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_34970*
_output_shapes
:*
dtype02 
AssignMovingAvg/ReadVariableOp�
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*(
_class
loc:@AssignMovingAvg/34970*
_output_shapes
:2
AssignMovingAvg/sub�
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*(
_class
loc:@AssignMovingAvg/34970*
_output_shapes
:2
AssignMovingAvg/mul�
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_34970AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*(
_class
loc:@AssignMovingAvg/34970*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp�
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:GPU:0**
_class 
loc:@AssignMovingAvg_1/34976*
_output_shapes
: *
dtype0*
valueB
 *
�#<2
AssignMovingAvg_1/decay�
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_34976*
_output_shapes
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOp�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0**
_class 
loc:@AssignMovingAvg_1/34976*
_output_shapes
:2
AssignMovingAvg_1/sub�
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0**
_class 
loc:@AssignMovingAvg_1/34976*
_output_shapes
:2
AssignMovingAvg_1/mul�
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_34976AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0**
_class 
loc:@AssignMovingAvg_1/34976*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:2
batchnorm/add/y�
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/Rsqrt�
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/mul/ReadVariableOp�
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mul�
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*3
_output_shapes!
:���������K2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2�
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp�
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/sub�
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*3
_output_shapes!
:���������K2
batchnorm/add_1�
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*3
_output_shapes!
:���������K2

Identity"
identityIdentity:output:0*B
_input_shapes1
/:���������K::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:[ W
3
_output_shapes!
:���������K
 
_user_specified_nameinputs
�
�
+__inference_normalize_2_layer_call_fn_36011

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *N
_output_shapes<
::8������������������������������������ *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*1J 8� *O
fJRH
F__inference_normalize_2_layer_call_and_return_conditional_losses_349182
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*N
_output_shapes<
::8������������������������������������ 2

Identity"
identityIdentity:output:0*]
_input_shapesL
J:8������������������������������������ ::::22
StatefulPartitionedCallStatefulPartitionedCall:v r
N
_output_shapes<
::8������������������������������������ 
 
_user_specified_nameinputs
�

�
#__inference_signature_wrapper_35487
input_7
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_7unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*2
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*1J 8� *)
f$R"
 __inference__wrapped_model_346492
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*s
_input_shapesb
`:���������==�::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:] Y
4
_output_shapes"
 :���������==�
!
_user_specified_name	input_7
�
�
+__inference_normalize_1_layer_call_fn_35909

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������K*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*1J 8� *O
fJRH
F__inference_normalize_1_layer_call_and_return_conditional_losses_350152
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*3
_output_shapes!
:���������K2

Identity"
identityIdentity:output:0*B
_input_shapes1
/:���������K::::22
StatefulPartitionedCallStatefulPartitionedCall:[ W
3
_output_shapes!
:���������K
 
_user_specified_nameinputs
�
^
B__inference_flatten_layer_call_and_return_conditional_losses_36119

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"������  2
Consti
ReshapeReshapeinputsConst:output:0*
T0*)
_output_shapes
:�����������2	
Reshapef
IdentityIdentityReshape:output:0*
T0*)
_output_shapes
:�����������2

Identity"
identityIdentity:output:0*2
_input_shapes!
:���������@:[ W
3
_output_shapes!
:���������@
 
_user_specified_nameinputs
�
�
F__inference_normalize_2_layer_call_and_return_conditional_losses_35133

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identity��batchnorm/ReadVariableOp�batchnorm/ReadVariableOp_1�batchnorm/ReadVariableOp_2�batchnorm/mul/ReadVariableOp�
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:2
batchnorm/add/y�
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
: 2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
: 2
batchnorm/Rsqrt�
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/mul/ReadVariableOp�
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 2
batchnorm/mul�
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*3
_output_shapes!
:���������% 2
batchnorm/mul_1�
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOp_1�
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
: 2
batchnorm/mul_2�
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOp_2�
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
: 2
batchnorm/sub�
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*3
_output_shapes!
:���������% 2
batchnorm/add_1�
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*3
_output_shapes!
:���������% 2

Identity"
identityIdentity:output:0*B
_input_shapes1
/:���������% ::::24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:[ W
3
_output_shapes!
:���������% 
 
_user_specified_nameinputs
�

�
'__inference_model_6_layer_call_fn_35725

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*2
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*1J 8� *K
fFRD
B__inference_model_6_layer_call_and_return_conditional_losses_354072
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*s
_input_shapesb
`:���������==�::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :���������==�
 
_user_specified_nameinputs
�

�
'__inference_model_6_layer_call_fn_35688

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*.
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*1J 8� *K
fFRD
B__inference_model_6_layer_call_and_return_conditional_losses_353272
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*s
_input_shapesb
`:���������==�::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :���������==�
 
_user_specified_nameinputs
�

�
C__inference_conv3d_2_layer_call_and_return_conditional_losses_35920

inputs"
conv3d_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�Conv3D/ReadVariableOp�
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
: *
dtype02
Conv3D/ReadVariableOp�
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������% *
paddingVALID*
strides	
2
Conv3D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������% 2	
BiasAddd
ReluReluBiasAdd:output:0*
T0*3
_output_shapes!
:���������% 2
Relu�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*
T0*3
_output_shapes!
:���������% 2

Identity"
identityIdentity:output:0*:
_input_shapes)
':���������K::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:[ W
3
_output_shapes!
:���������K
 
_user_specified_nameinputs
�
�
F__inference_normalize_2_layer_call_and_return_conditional_losses_36067

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identity��batchnorm/ReadVariableOp�batchnorm/ReadVariableOp_1�batchnorm/ReadVariableOp_2�batchnorm/mul/ReadVariableOp�
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:2
batchnorm/add/y�
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
: 2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
: 2
batchnorm/Rsqrt�
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/mul/ReadVariableOp�
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 2
batchnorm/mul�
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*3
_output_shapes!
:���������% 2
batchnorm/mul_1�
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOp_1�
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
: 2
batchnorm/mul_2�
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOp_2�
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
: 2
batchnorm/sub�
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*3
_output_shapes!
:���������% 2
batchnorm/add_1�
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*3
_output_shapes!
:���������% 2

Identity"
identityIdentity:output:0*B
_input_shapes1
/:���������% ::::24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:[ W
3
_output_shapes!
:���������% 
 
_user_specified_nameinputs
�
�
+__inference_normalize_2_layer_call_fn_36093

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������% *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*1J 8� *O
fJRH
F__inference_normalize_2_layer_call_and_return_conditional_losses_351332
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*3
_output_shapes!
:���������% 2

Identity"
identityIdentity:output:0*B
_input_shapes1
/:���������% ::::22
StatefulPartitionedCallStatefulPartitionedCall:[ W
3
_output_shapes!
:���������% 
 
_user_specified_nameinputs
�h
�
 __inference__wrapped_model_34649
input_73
/model_6_conv3d_1_conv3d_readvariableop_resource4
0model_6_conv3d_1_biasadd_readvariableop_resource9
5model_6_normalize_1_batchnorm_readvariableop_resource=
9model_6_normalize_1_batchnorm_mul_readvariableop_resource;
7model_6_normalize_1_batchnorm_readvariableop_1_resource;
7model_6_normalize_1_batchnorm_readvariableop_2_resource3
/model_6_conv3d_2_conv3d_readvariableop_resource4
0model_6_conv3d_2_biasadd_readvariableop_resource9
5model_6_normalize_2_batchnorm_readvariableop_resource=
9model_6_normalize_2_batchnorm_mul_readvariableop_resource;
7model_6_normalize_2_batchnorm_readvariableop_1_resource;
7model_6_normalize_2_batchnorm_readvariableop_2_resource3
/model_6_conv3d_3_conv3d_readvariableop_resource4
0model_6_conv3d_3_biasadd_readvariableop_resource0
,model_6_theta_matmul_readvariableop_resource1
-model_6_theta_biasadd_readvariableop_resource
identity��'model_6/conv3d_1/BiasAdd/ReadVariableOp�&model_6/conv3d_1/Conv3D/ReadVariableOp�'model_6/conv3d_2/BiasAdd/ReadVariableOp�&model_6/conv3d_2/Conv3D/ReadVariableOp�'model_6/conv3d_3/BiasAdd/ReadVariableOp�&model_6/conv3d_3/Conv3D/ReadVariableOp�,model_6/normalize_1/batchnorm/ReadVariableOp�.model_6/normalize_1/batchnorm/ReadVariableOp_1�.model_6/normalize_1/batchnorm/ReadVariableOp_2�0model_6/normalize_1/batchnorm/mul/ReadVariableOp�,model_6/normalize_2/batchnorm/ReadVariableOp�.model_6/normalize_2/batchnorm/ReadVariableOp_1�.model_6/normalize_2/batchnorm/ReadVariableOp_2�0model_6/normalize_2/batchnorm/mul/ReadVariableOp�$model_6/theta/BiasAdd/ReadVariableOp�#model_6/theta/MatMul/ReadVariableOp�
&model_6/conv3d_1/Conv3D/ReadVariableOpReadVariableOp/model_6_conv3d_1_conv3d_readvariableop_resource**
_output_shapes
:*
dtype02(
&model_6/conv3d_1/Conv3D/ReadVariableOp�
model_6/conv3d_1/Conv3DConv3Dinput_7.model_6/conv3d_1/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������K*
paddingVALID*
strides	
2
model_6/conv3d_1/Conv3D�
'model_6/conv3d_1/BiasAdd/ReadVariableOpReadVariableOp0model_6_conv3d_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02)
'model_6/conv3d_1/BiasAdd/ReadVariableOp�
model_6/conv3d_1/BiasAddBiasAdd model_6/conv3d_1/Conv3D:output:0/model_6/conv3d_1/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������K2
model_6/conv3d_1/BiasAdd�
model_6/conv3d_1/ReluRelu!model_6/conv3d_1/BiasAdd:output:0*
T0*3
_output_shapes!
:���������K2
model_6/conv3d_1/Relu�
,model_6/normalize_1/batchnorm/ReadVariableOpReadVariableOp5model_6_normalize_1_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02.
,model_6/normalize_1/batchnorm/ReadVariableOp�
#model_6/normalize_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:2%
#model_6/normalize_1/batchnorm/add/y�
!model_6/normalize_1/batchnorm/addAddV24model_6/normalize_1/batchnorm/ReadVariableOp:value:0,model_6/normalize_1/batchnorm/add/y:output:0*
T0*
_output_shapes
:2#
!model_6/normalize_1/batchnorm/add�
#model_6/normalize_1/batchnorm/RsqrtRsqrt%model_6/normalize_1/batchnorm/add:z:0*
T0*
_output_shapes
:2%
#model_6/normalize_1/batchnorm/Rsqrt�
0model_6/normalize_1/batchnorm/mul/ReadVariableOpReadVariableOp9model_6_normalize_1_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype022
0model_6/normalize_1/batchnorm/mul/ReadVariableOp�
!model_6/normalize_1/batchnorm/mulMul'model_6/normalize_1/batchnorm/Rsqrt:y:08model_6/normalize_1/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2#
!model_6/normalize_1/batchnorm/mul�
#model_6/normalize_1/batchnorm/mul_1Mul#model_6/conv3d_1/Relu:activations:0%model_6/normalize_1/batchnorm/mul:z:0*
T0*3
_output_shapes!
:���������K2%
#model_6/normalize_1/batchnorm/mul_1�
.model_6/normalize_1/batchnorm/ReadVariableOp_1ReadVariableOp7model_6_normalize_1_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype020
.model_6/normalize_1/batchnorm/ReadVariableOp_1�
#model_6/normalize_1/batchnorm/mul_2Mul6model_6/normalize_1/batchnorm/ReadVariableOp_1:value:0%model_6/normalize_1/batchnorm/mul:z:0*
T0*
_output_shapes
:2%
#model_6/normalize_1/batchnorm/mul_2�
.model_6/normalize_1/batchnorm/ReadVariableOp_2ReadVariableOp7model_6_normalize_1_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype020
.model_6/normalize_1/batchnorm/ReadVariableOp_2�
!model_6/normalize_1/batchnorm/subSub6model_6/normalize_1/batchnorm/ReadVariableOp_2:value:0'model_6/normalize_1/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2#
!model_6/normalize_1/batchnorm/sub�
#model_6/normalize_1/batchnorm/add_1AddV2'model_6/normalize_1/batchnorm/mul_1:z:0%model_6/normalize_1/batchnorm/sub:z:0*
T0*3
_output_shapes!
:���������K2%
#model_6/normalize_1/batchnorm/add_1�
&model_6/conv3d_2/Conv3D/ReadVariableOpReadVariableOp/model_6_conv3d_2_conv3d_readvariableop_resource**
_output_shapes
: *
dtype02(
&model_6/conv3d_2/Conv3D/ReadVariableOp�
model_6/conv3d_2/Conv3DConv3D'model_6/normalize_1/batchnorm/add_1:z:0.model_6/conv3d_2/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������% *
paddingVALID*
strides	
2
model_6/conv3d_2/Conv3D�
'model_6/conv3d_2/BiasAdd/ReadVariableOpReadVariableOp0model_6_conv3d_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02)
'model_6/conv3d_2/BiasAdd/ReadVariableOp�
model_6/conv3d_2/BiasAddBiasAdd model_6/conv3d_2/Conv3D:output:0/model_6/conv3d_2/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������% 2
model_6/conv3d_2/BiasAdd�
model_6/conv3d_2/ReluRelu!model_6/conv3d_2/BiasAdd:output:0*
T0*3
_output_shapes!
:���������% 2
model_6/conv3d_2/Relu�
,model_6/normalize_2/batchnorm/ReadVariableOpReadVariableOp5model_6_normalize_2_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype02.
,model_6/normalize_2/batchnorm/ReadVariableOp�
#model_6/normalize_2/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:2%
#model_6/normalize_2/batchnorm/add/y�
!model_6/normalize_2/batchnorm/addAddV24model_6/normalize_2/batchnorm/ReadVariableOp:value:0,model_6/normalize_2/batchnorm/add/y:output:0*
T0*
_output_shapes
: 2#
!model_6/normalize_2/batchnorm/add�
#model_6/normalize_2/batchnorm/RsqrtRsqrt%model_6/normalize_2/batchnorm/add:z:0*
T0*
_output_shapes
: 2%
#model_6/normalize_2/batchnorm/Rsqrt�
0model_6/normalize_2/batchnorm/mul/ReadVariableOpReadVariableOp9model_6_normalize_2_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype022
0model_6/normalize_2/batchnorm/mul/ReadVariableOp�
!model_6/normalize_2/batchnorm/mulMul'model_6/normalize_2/batchnorm/Rsqrt:y:08model_6/normalize_2/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 2#
!model_6/normalize_2/batchnorm/mul�
#model_6/normalize_2/batchnorm/mul_1Mul#model_6/conv3d_2/Relu:activations:0%model_6/normalize_2/batchnorm/mul:z:0*
T0*3
_output_shapes!
:���������% 2%
#model_6/normalize_2/batchnorm/mul_1�
.model_6/normalize_2/batchnorm/ReadVariableOp_1ReadVariableOp7model_6_normalize_2_batchnorm_readvariableop_1_resource*
_output_shapes
: *
dtype020
.model_6/normalize_2/batchnorm/ReadVariableOp_1�
#model_6/normalize_2/batchnorm/mul_2Mul6model_6/normalize_2/batchnorm/ReadVariableOp_1:value:0%model_6/normalize_2/batchnorm/mul:z:0*
T0*
_output_shapes
: 2%
#model_6/normalize_2/batchnorm/mul_2�
.model_6/normalize_2/batchnorm/ReadVariableOp_2ReadVariableOp7model_6_normalize_2_batchnorm_readvariableop_2_resource*
_output_shapes
: *
dtype020
.model_6/normalize_2/batchnorm/ReadVariableOp_2�
!model_6/normalize_2/batchnorm/subSub6model_6/normalize_2/batchnorm/ReadVariableOp_2:value:0'model_6/normalize_2/batchnorm/mul_2:z:0*
T0*
_output_shapes
: 2#
!model_6/normalize_2/batchnorm/sub�
#model_6/normalize_2/batchnorm/add_1AddV2'model_6/normalize_2/batchnorm/mul_1:z:0%model_6/normalize_2/batchnorm/sub:z:0*
T0*3
_output_shapes!
:���������% 2%
#model_6/normalize_2/batchnorm/add_1�
&model_6/conv3d_3/Conv3D/ReadVariableOpReadVariableOp/model_6_conv3d_3_conv3d_readvariableop_resource**
_output_shapes
: @*
dtype02(
&model_6/conv3d_3/Conv3D/ReadVariableOp�
model_6/conv3d_3/Conv3DConv3D'model_6/normalize_2/batchnorm/add_1:z:0.model_6/conv3d_3/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������@*
paddingVALID*
strides	
2
model_6/conv3d_3/Conv3D�
'model_6/conv3d_3/BiasAdd/ReadVariableOpReadVariableOp0model_6_conv3d_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02)
'model_6/conv3d_3/BiasAdd/ReadVariableOp�
model_6/conv3d_3/BiasAddBiasAdd model_6/conv3d_3/Conv3D:output:0/model_6/conv3d_3/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������@2
model_6/conv3d_3/BiasAdd�
model_6/conv3d_3/ReluRelu!model_6/conv3d_3/BiasAdd:output:0*
T0*3
_output_shapes!
:���������@2
model_6/conv3d_3/Relu
model_6/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"������  2
model_6/flatten/Const�
model_6/flatten/ReshapeReshape#model_6/conv3d_3/Relu:activations:0model_6/flatten/Const:output:0*
T0*)
_output_shapes
:�����������2
model_6/flatten/Reshape�
#model_6/theta/MatMul/ReadVariableOpReadVariableOp,model_6_theta_matmul_readvariableop_resource*!
_output_shapes
:���*
dtype02%
#model_6/theta/MatMul/ReadVariableOp�
model_6/theta/MatMulMatMul model_6/flatten/Reshape:output:0+model_6/theta/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
model_6/theta/MatMul�
$model_6/theta/BiasAdd/ReadVariableOpReadVariableOp-model_6_theta_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02&
$model_6/theta/BiasAdd/ReadVariableOp�
model_6/theta/BiasAddBiasAddmodel_6/theta/MatMul:product:0,model_6/theta/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
model_6/theta/BiasAdd�
model_6/theta/SoftmaxSoftmaxmodel_6/theta/BiasAdd:output:0*
T0*(
_output_shapes
:����������2
model_6/theta/Softmax�
IdentityIdentitymodel_6/theta/Softmax:softmax:0(^model_6/conv3d_1/BiasAdd/ReadVariableOp'^model_6/conv3d_1/Conv3D/ReadVariableOp(^model_6/conv3d_2/BiasAdd/ReadVariableOp'^model_6/conv3d_2/Conv3D/ReadVariableOp(^model_6/conv3d_3/BiasAdd/ReadVariableOp'^model_6/conv3d_3/Conv3D/ReadVariableOp-^model_6/normalize_1/batchnorm/ReadVariableOp/^model_6/normalize_1/batchnorm/ReadVariableOp_1/^model_6/normalize_1/batchnorm/ReadVariableOp_21^model_6/normalize_1/batchnorm/mul/ReadVariableOp-^model_6/normalize_2/batchnorm/ReadVariableOp/^model_6/normalize_2/batchnorm/ReadVariableOp_1/^model_6/normalize_2/batchnorm/ReadVariableOp_21^model_6/normalize_2/batchnorm/mul/ReadVariableOp%^model_6/theta/BiasAdd/ReadVariableOp$^model_6/theta/MatMul/ReadVariableOp*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*s
_input_shapesb
`:���������==�::::::::::::::::2R
'model_6/conv3d_1/BiasAdd/ReadVariableOp'model_6/conv3d_1/BiasAdd/ReadVariableOp2P
&model_6/conv3d_1/Conv3D/ReadVariableOp&model_6/conv3d_1/Conv3D/ReadVariableOp2R
'model_6/conv3d_2/BiasAdd/ReadVariableOp'model_6/conv3d_2/BiasAdd/ReadVariableOp2P
&model_6/conv3d_2/Conv3D/ReadVariableOp&model_6/conv3d_2/Conv3D/ReadVariableOp2R
'model_6/conv3d_3/BiasAdd/ReadVariableOp'model_6/conv3d_3/BiasAdd/ReadVariableOp2P
&model_6/conv3d_3/Conv3D/ReadVariableOp&model_6/conv3d_3/Conv3D/ReadVariableOp2\
,model_6/normalize_1/batchnorm/ReadVariableOp,model_6/normalize_1/batchnorm/ReadVariableOp2`
.model_6/normalize_1/batchnorm/ReadVariableOp_1.model_6/normalize_1/batchnorm/ReadVariableOp_12`
.model_6/normalize_1/batchnorm/ReadVariableOp_2.model_6/normalize_1/batchnorm/ReadVariableOp_22d
0model_6/normalize_1/batchnorm/mul/ReadVariableOp0model_6/normalize_1/batchnorm/mul/ReadVariableOp2\
,model_6/normalize_2/batchnorm/ReadVariableOp,model_6/normalize_2/batchnorm/ReadVariableOp2`
.model_6/normalize_2/batchnorm/ReadVariableOp_1.model_6/normalize_2/batchnorm/ReadVariableOp_12`
.model_6/normalize_2/batchnorm/ReadVariableOp_2.model_6/normalize_2/batchnorm/ReadVariableOp_22d
0model_6/normalize_2/batchnorm/mul/ReadVariableOp0model_6/normalize_2/batchnorm/mul/ReadVariableOp2L
$model_6/theta/BiasAdd/ReadVariableOp$model_6/theta/BiasAdd/ReadVariableOp2J
#model_6/theta/MatMul/ReadVariableOp#model_6/theta/MatMul/ReadVariableOp:] Y
4
_output_shapes"
 :���������==�
!
_user_specified_name	input_7
�%
�
B__inference_model_6_layer_call_and_return_conditional_losses_35281
input_7
conv3d_1_35241
conv3d_1_35243
normalize_1_35246
normalize_1_35248
normalize_1_35250
normalize_1_35252
conv3d_2_35255
conv3d_2_35257
normalize_2_35260
normalize_2_35262
normalize_2_35264
normalize_2_35266
conv3d_3_35269
conv3d_3_35271
theta_35275
theta_35277
identity�� conv3d_1/StatefulPartitionedCall� conv3d_2/StatefulPartitionedCall� conv3d_3/StatefulPartitionedCall�#normalize_1/StatefulPartitionedCall�#normalize_2/StatefulPartitionedCall�theta/StatefulPartitionedCall�
 conv3d_1/StatefulPartitionedCallStatefulPartitionedCallinput_7conv3d_1_35241conv3d_1_35243*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������K*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*1J 8� *L
fGRE
C__inference_conv3d_1_layer_call_and_return_conditional_losses_349442"
 conv3d_1/StatefulPartitionedCall�
#normalize_1/StatefulPartitionedCallStatefulPartitionedCall)conv3d_1/StatefulPartitionedCall:output:0normalize_1_35246normalize_1_35248normalize_1_35250normalize_1_35252*
Tin	
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������K*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*1J 8� *O
fJRH
F__inference_normalize_1_layer_call_and_return_conditional_losses_350152%
#normalize_1/StatefulPartitionedCall�
 conv3d_2/StatefulPartitionedCallStatefulPartitionedCall,normalize_1/StatefulPartitionedCall:output:0conv3d_2_35255conv3d_2_35257*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������% *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*1J 8� *L
fGRE
C__inference_conv3d_2_layer_call_and_return_conditional_losses_350622"
 conv3d_2/StatefulPartitionedCall�
#normalize_2/StatefulPartitionedCallStatefulPartitionedCall)conv3d_2/StatefulPartitionedCall:output:0normalize_2_35260normalize_2_35262normalize_2_35264normalize_2_35266*
Tin	
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������% *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*1J 8� *O
fJRH
F__inference_normalize_2_layer_call_and_return_conditional_losses_351332%
#normalize_2/StatefulPartitionedCall�
 conv3d_3/StatefulPartitionedCallStatefulPartitionedCall,normalize_2/StatefulPartitionedCall:output:0conv3d_3_35269conv3d_3_35271*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*1J 8� *L
fGRE
C__inference_conv3d_3_layer_call_and_return_conditional_losses_351802"
 conv3d_3/StatefulPartitionedCall�
flatten/PartitionedCallPartitionedCall)conv3d_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:�����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*1J 8� *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_352022
flatten/PartitionedCall�
theta/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0theta_35275theta_35277*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*1J 8� *I
fDRB
@__inference_theta_layer_call_and_return_conditional_losses_352212
theta/StatefulPartitionedCall�
IdentityIdentity&theta/StatefulPartitionedCall:output:0!^conv3d_1/StatefulPartitionedCall!^conv3d_2/StatefulPartitionedCall!^conv3d_3/StatefulPartitionedCall$^normalize_1/StatefulPartitionedCall$^normalize_2/StatefulPartitionedCall^theta/StatefulPartitionedCall*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*s
_input_shapesb
`:���������==�::::::::::::::::2D
 conv3d_1/StatefulPartitionedCall conv3d_1/StatefulPartitionedCall2D
 conv3d_2/StatefulPartitionedCall conv3d_2/StatefulPartitionedCall2D
 conv3d_3/StatefulPartitionedCall conv3d_3/StatefulPartitionedCall2J
#normalize_1/StatefulPartitionedCall#normalize_1/StatefulPartitionedCall2J
#normalize_2/StatefulPartitionedCall#normalize_2/StatefulPartitionedCall2>
theta/StatefulPartitionedCalltheta/StatefulPartitionedCall:] Y
4
_output_shapes"
 :���������==�
!
_user_specified_name	input_7
�

�
C__inference_conv3d_1_layer_call_and_return_conditional_losses_35736

inputs"
conv3d_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�Conv3D/ReadVariableOp�
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:*
dtype02
Conv3D/ReadVariableOp�
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������K*
paddingVALID*
strides	
2
Conv3D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������K2	
BiasAddd
ReluReluBiasAdd:output:0*
T0*3
_output_shapes!
:���������K2
Relu�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*
T0*3
_output_shapes!
:���������K2

Identity"
identityIdentity:output:0*;
_input_shapes*
(:���������==�::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:\ X
4
_output_shapes"
 :���������==�
 
_user_specified_nameinputs
�%
�
B__inference_model_6_layer_call_and_return_conditional_losses_35327

inputs
conv3d_1_35287
conv3d_1_35289
normalize_1_35292
normalize_1_35294
normalize_1_35296
normalize_1_35298
conv3d_2_35301
conv3d_2_35303
normalize_2_35306
normalize_2_35308
normalize_2_35310
normalize_2_35312
conv3d_3_35315
conv3d_3_35317
theta_35321
theta_35323
identity�� conv3d_1/StatefulPartitionedCall� conv3d_2/StatefulPartitionedCall� conv3d_3/StatefulPartitionedCall�#normalize_1/StatefulPartitionedCall�#normalize_2/StatefulPartitionedCall�theta/StatefulPartitionedCall�
 conv3d_1/StatefulPartitionedCallStatefulPartitionedCallinputsconv3d_1_35287conv3d_1_35289*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������K*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*1J 8� *L
fGRE
C__inference_conv3d_1_layer_call_and_return_conditional_losses_349442"
 conv3d_1/StatefulPartitionedCall�
#normalize_1/StatefulPartitionedCallStatefulPartitionedCall)conv3d_1/StatefulPartitionedCall:output:0normalize_1_35292normalize_1_35294normalize_1_35296normalize_1_35298*
Tin	
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������K*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*1J 8� *O
fJRH
F__inference_normalize_1_layer_call_and_return_conditional_losses_349952%
#normalize_1/StatefulPartitionedCall�
 conv3d_2/StatefulPartitionedCallStatefulPartitionedCall,normalize_1/StatefulPartitionedCall:output:0conv3d_2_35301conv3d_2_35303*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������% *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*1J 8� *L
fGRE
C__inference_conv3d_2_layer_call_and_return_conditional_losses_350622"
 conv3d_2/StatefulPartitionedCall�
#normalize_2/StatefulPartitionedCallStatefulPartitionedCall)conv3d_2/StatefulPartitionedCall:output:0normalize_2_35306normalize_2_35308normalize_2_35310normalize_2_35312*
Tin	
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������% *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*1J 8� *O
fJRH
F__inference_normalize_2_layer_call_and_return_conditional_losses_351132%
#normalize_2/StatefulPartitionedCall�
 conv3d_3/StatefulPartitionedCallStatefulPartitionedCall,normalize_2/StatefulPartitionedCall:output:0conv3d_3_35315conv3d_3_35317*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*1J 8� *L
fGRE
C__inference_conv3d_3_layer_call_and_return_conditional_losses_351802"
 conv3d_3/StatefulPartitionedCall�
flatten/PartitionedCallPartitionedCall)conv3d_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:�����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*1J 8� *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_352022
flatten/PartitionedCall�
theta/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0theta_35321theta_35323*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*1J 8� *I
fDRB
@__inference_theta_layer_call_and_return_conditional_losses_352212
theta/StatefulPartitionedCall�
IdentityIdentity&theta/StatefulPartitionedCall:output:0!^conv3d_1/StatefulPartitionedCall!^conv3d_2/StatefulPartitionedCall!^conv3d_3/StatefulPartitionedCall$^normalize_1/StatefulPartitionedCall$^normalize_2/StatefulPartitionedCall^theta/StatefulPartitionedCall*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*s
_input_shapesb
`:���������==�::::::::::::::::2D
 conv3d_1/StatefulPartitionedCall conv3d_1/StatefulPartitionedCall2D
 conv3d_2/StatefulPartitionedCall conv3d_2/StatefulPartitionedCall2D
 conv3d_3/StatefulPartitionedCall conv3d_3/StatefulPartitionedCall2J
#normalize_1/StatefulPartitionedCall#normalize_1/StatefulPartitionedCall2J
#normalize_2/StatefulPartitionedCall#normalize_2/StatefulPartitionedCall2>
theta/StatefulPartitionedCalltheta/StatefulPartitionedCall:\ X
4
_output_shapes"
 :���������==�
 
_user_specified_nameinputs
�1
�
F__inference_normalize_1_layer_call_and_return_conditional_losses_35863

inputs
assignmovingavg_35838
assignmovingavg_1_35844)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identity��#AssignMovingAvg/AssignSubVariableOp�AssignMovingAvg/ReadVariableOp�%AssignMovingAvg_1/AssignSubVariableOp� AssignMovingAvg_1/ReadVariableOp�batchnorm/ReadVariableOp�batchnorm/mul/ReadVariableOp�
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*%
valueB"             2 
moments/mean/reduction_indices�
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0**
_output_shapes
:*
	keep_dims(2
moments/mean�
moments/StopGradientStopGradientmoments/mean:output:0*
T0**
_output_shapes
:2
moments/StopGradient�
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*3
_output_shapes!
:���������K2
moments/SquaredDifference�
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"moments/variance/reduction_indices�
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0**
_output_shapes
:*
	keep_dims(2
moments/variance�
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze�
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze_1�
AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:GPU:0*(
_class
loc:@AssignMovingAvg/35838*
_output_shapes
: *
dtype0*
valueB
 *
�#<2
AssignMovingAvg/decay�
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_35838*
_output_shapes
:*
dtype02 
AssignMovingAvg/ReadVariableOp�
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*(
_class
loc:@AssignMovingAvg/35838*
_output_shapes
:2
AssignMovingAvg/sub�
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*(
_class
loc:@AssignMovingAvg/35838*
_output_shapes
:2
AssignMovingAvg/mul�
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_35838AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*(
_class
loc:@AssignMovingAvg/35838*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp�
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:GPU:0**
_class 
loc:@AssignMovingAvg_1/35844*
_output_shapes
: *
dtype0*
valueB
 *
�#<2
AssignMovingAvg_1/decay�
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_35844*
_output_shapes
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOp�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0**
_class 
loc:@AssignMovingAvg_1/35844*
_output_shapes
:2
AssignMovingAvg_1/sub�
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0**
_class 
loc:@AssignMovingAvg_1/35844*
_output_shapes
:2
AssignMovingAvg_1/mul�
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_35844AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0**
_class 
loc:@AssignMovingAvg_1/35844*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:2
batchnorm/add/y�
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/Rsqrt�
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/mul/ReadVariableOp�
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mul�
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*3
_output_shapes!
:���������K2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2�
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp�
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/sub�
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*3
_output_shapes!
:���������K2
batchnorm/add_1�
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*3
_output_shapes!
:���������K2

Identity"
identityIdentity:output:0*B
_input_shapes1
/:���������K::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:[ W
3
_output_shapes!
:���������K
 
_user_specified_nameinputs
�
�
F__inference_normalize_1_layer_call_and_return_conditional_losses_34778

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identity��batchnorm/ReadVariableOp�batchnorm/ReadVariableOp_1�batchnorm/ReadVariableOp_2�batchnorm/mul/ReadVariableOp�
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:2
batchnorm/add/y�
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/Rsqrt�
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/mul/ReadVariableOp�
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mul�
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*N
_output_shapes<
::8������������������������������������2
batchnorm/mul_1�
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp_1�
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2�
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp_2�
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/sub�
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*N
_output_shapes<
::8������������������������������������2
batchnorm/add_1�
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*N
_output_shapes<
::8������������������������������������2

Identity"
identityIdentity:output:0*]
_input_shapesL
J:8������������������������������������::::24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:v r
N
_output_shapes<
::8������������������������������������
 
_user_specified_nameinputs
�3
�	
__inference__traced_save_36233
file_prefix.
*savev2_conv3d_1_kernel_read_readvariableop,
(savev2_conv3d_1_bias_read_readvariableop0
,savev2_normalize_1_gamma_read_readvariableop/
+savev2_normalize_1_beta_read_readvariableop6
2savev2_normalize_1_moving_mean_read_readvariableop:
6savev2_normalize_1_moving_variance_read_readvariableop.
*savev2_conv3d_2_kernel_read_readvariableop,
(savev2_conv3d_2_bias_read_readvariableop0
,savev2_normalize_2_gamma_read_readvariableop/
+savev2_normalize_2_beta_read_readvariableop6
2savev2_normalize_2_moving_mean_read_readvariableop:
6savev2_normalize_2_moving_variance_read_readvariableop.
*savev2_conv3d_3_kernel_read_readvariableop,
(savev2_conv3d_3_bias_read_readvariableop+
'savev2_theta_kernel_read_readvariableop)
%savev2_theta_bias_read_readvariableop'
#savev2_sgd_iter_read_readvariableop	(
$savev2_sgd_decay_read_readvariableop0
,savev2_sgd_learning_rate_read_readvariableop+
'savev2_sgd_momentum_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop
savev2_const

identity_1��MergeV2Checkpoints�
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1�
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard�
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename�

SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�	
value�	B�	B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*A
value8B6B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices�	
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0*savev2_conv3d_1_kernel_read_readvariableop(savev2_conv3d_1_bias_read_readvariableop,savev2_normalize_1_gamma_read_readvariableop+savev2_normalize_1_beta_read_readvariableop2savev2_normalize_1_moving_mean_read_readvariableop6savev2_normalize_1_moving_variance_read_readvariableop*savev2_conv3d_2_kernel_read_readvariableop(savev2_conv3d_2_bias_read_readvariableop,savev2_normalize_2_gamma_read_readvariableop+savev2_normalize_2_beta_read_readvariableop2savev2_normalize_2_moving_mean_read_readvariableop6savev2_normalize_2_moving_variance_read_readvariableop*savev2_conv3d_3_kernel_read_readvariableop(savev2_conv3d_3_bias_read_readvariableop'savev2_theta_kernel_read_readvariableop%savev2_theta_bias_read_readvariableop#savev2_sgd_iter_read_readvariableop$savev2_sgd_decay_read_readvariableop,savev2_sgd_learning_rate_read_readvariableop'savev2_sgd_momentum_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *%
dtypes
2	2
SaveV2�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*�
_input_shapes�
�: ::::::: : : : : : : @:@:���:�: : : : : : : 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:0,
*
_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::0,
*
_output_shapes
: : 

_output_shapes
: : 	

_output_shapes
: : 


_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: :0,
*
_output_shapes
: @: 

_output_shapes
:@:'#
!
_output_shapes
:���:!

_output_shapes	
:�:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
�\
�
!__inference__traced_restore_36309
file_prefix$
 assignvariableop_conv3d_1_kernel$
 assignvariableop_1_conv3d_1_bias(
$assignvariableop_2_normalize_1_gamma'
#assignvariableop_3_normalize_1_beta.
*assignvariableop_4_normalize_1_moving_mean2
.assignvariableop_5_normalize_1_moving_variance&
"assignvariableop_6_conv3d_2_kernel$
 assignvariableop_7_conv3d_2_bias(
$assignvariableop_8_normalize_2_gamma'
#assignvariableop_9_normalize_2_beta/
+assignvariableop_10_normalize_2_moving_mean3
/assignvariableop_11_normalize_2_moving_variance'
#assignvariableop_12_conv3d_3_kernel%
!assignvariableop_13_conv3d_3_bias$
 assignvariableop_14_theta_kernel"
assignvariableop_15_theta_bias 
assignvariableop_16_sgd_iter!
assignvariableop_17_sgd_decay)
%assignvariableop_18_sgd_learning_rate$
 assignvariableop_19_sgd_momentum
assignvariableop_20_total
assignvariableop_21_count
identity_23��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_18�AssignVariableOp_19�AssignVariableOp_2�AssignVariableOp_20�AssignVariableOp_21�AssignVariableOp_3�AssignVariableOp_4�AssignVariableOp_5�AssignVariableOp_6�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�

RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�	
value�	B�	B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*A
value8B6B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices�
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*p
_output_shapes^
\:::::::::::::::::::::::*%
dtypes
2	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity�
AssignVariableOpAssignVariableOp assignvariableop_conv3d_1_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1�
AssignVariableOp_1AssignVariableOp assignvariableop_1_conv3d_1_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2�
AssignVariableOp_2AssignVariableOp$assignvariableop_2_normalize_1_gammaIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3�
AssignVariableOp_3AssignVariableOp#assignvariableop_3_normalize_1_betaIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4�
AssignVariableOp_4AssignVariableOp*assignvariableop_4_normalize_1_moving_meanIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5�
AssignVariableOp_5AssignVariableOp.assignvariableop_5_normalize_1_moving_varianceIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6�
AssignVariableOp_6AssignVariableOp"assignvariableop_6_conv3d_2_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7�
AssignVariableOp_7AssignVariableOp assignvariableop_7_conv3d_2_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8�
AssignVariableOp_8AssignVariableOp$assignvariableop_8_normalize_2_gammaIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9�
AssignVariableOp_9AssignVariableOp#assignvariableop_9_normalize_2_betaIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10�
AssignVariableOp_10AssignVariableOp+assignvariableop_10_normalize_2_moving_meanIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11�
AssignVariableOp_11AssignVariableOp/assignvariableop_11_normalize_2_moving_varianceIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12�
AssignVariableOp_12AssignVariableOp#assignvariableop_12_conv3d_3_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13�
AssignVariableOp_13AssignVariableOp!assignvariableop_13_conv3d_3_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14�
AssignVariableOp_14AssignVariableOp assignvariableop_14_theta_kernelIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15�
AssignVariableOp_15AssignVariableOpassignvariableop_15_theta_biasIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0	*
_output_shapes
:2
Identity_16�
AssignVariableOp_16AssignVariableOpassignvariableop_16_sgd_iterIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17�
AssignVariableOp_17AssignVariableOpassignvariableop_17_sgd_decayIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18�
AssignVariableOp_18AssignVariableOp%assignvariableop_18_sgd_learning_rateIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19�
AssignVariableOp_19AssignVariableOp assignvariableop_19_sgd_momentumIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20�
AssignVariableOp_20AssignVariableOpassignvariableop_20_totalIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21�
AssignVariableOp_21AssignVariableOpassignvariableop_21_countIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_219
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp�
Identity_22Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_22�
Identity_23IdentityIdentity_22:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_23"#
identity_23Identity_23:output:0*m
_input_shapes\
Z: ::::::::::::::::::::::2$
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
AssignVariableOp_21AssignVariableOp_212(
AssignVariableOp_3AssignVariableOp_32(
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
_user_specified_namefile_prefix
�

�
'__inference_model_6_layer_call_fn_35442
input_7
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_7unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*2
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*1J 8� *K
fFRD
B__inference_model_6_layer_call_and_return_conditional_losses_354072
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*s
_input_shapesb
`:���������==�::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:] Y
4
_output_shapes"
 :���������==�
!
_user_specified_name	input_7
�
�
F__inference_normalize_2_layer_call_and_return_conditional_losses_35985

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identity��batchnorm/ReadVariableOp�batchnorm/ReadVariableOp_1�batchnorm/ReadVariableOp_2�batchnorm/mul/ReadVariableOp�
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:2
batchnorm/add/y�
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
: 2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
: 2
batchnorm/Rsqrt�
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/mul/ReadVariableOp�
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 2
batchnorm/mul�
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*N
_output_shapes<
::8������������������������������������ 2
batchnorm/mul_1�
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOp_1�
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
: 2
batchnorm/mul_2�
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOp_2�
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
: 2
batchnorm/sub�
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*N
_output_shapes<
::8������������������������������������ 2
batchnorm/add_1�
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*N
_output_shapes<
::8������������������������������������ 2

Identity"
identityIdentity:output:0*]
_input_shapesL
J:8������������������������������������ ::::24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:v r
N
_output_shapes<
::8������������������������������������ 
 
_user_specified_nameinputs
�
�
F__inference_normalize_1_layer_call_and_return_conditional_losses_35801

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identity��batchnorm/ReadVariableOp�batchnorm/ReadVariableOp_1�batchnorm/ReadVariableOp_2�batchnorm/mul/ReadVariableOp�
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:2
batchnorm/add/y�
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/Rsqrt�
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/mul/ReadVariableOp�
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mul�
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*N
_output_shapes<
::8������������������������������������2
batchnorm/mul_1�
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp_1�
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2�
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp_2�
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/sub�
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*N
_output_shapes<
::8������������������������������������2
batchnorm/add_1�
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*N
_output_shapes<
::8������������������������������������2

Identity"
identityIdentity:output:0*]
_input_shapesL
J:8������������������������������������::::24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:v r
N
_output_shapes<
::8������������������������������������
 
_user_specified_nameinputs
�2
�
F__inference_normalize_1_layer_call_and_return_conditional_losses_34745

inputs
assignmovingavg_34720
assignmovingavg_1_34726)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identity��#AssignMovingAvg/AssignSubVariableOp�AssignMovingAvg/ReadVariableOp�%AssignMovingAvg_1/AssignSubVariableOp� AssignMovingAvg_1/ReadVariableOp�batchnorm/ReadVariableOp�batchnorm/mul/ReadVariableOp�
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*%
valueB"             2 
moments/mean/reduction_indices�
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0**
_output_shapes
:*
	keep_dims(2
moments/mean�
moments/StopGradientStopGradientmoments/mean:output:0*
T0**
_output_shapes
:2
moments/StopGradient�
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*N
_output_shapes<
::8������������������������������������2
moments/SquaredDifference�
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"moments/variance/reduction_indices�
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0**
_output_shapes
:*
	keep_dims(2
moments/variance�
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze�
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze_1�
AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:GPU:0*(
_class
loc:@AssignMovingAvg/34720*
_output_shapes
: *
dtype0*
valueB
 *
�#<2
AssignMovingAvg/decay�
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_34720*
_output_shapes
:*
dtype02 
AssignMovingAvg/ReadVariableOp�
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*(
_class
loc:@AssignMovingAvg/34720*
_output_shapes
:2
AssignMovingAvg/sub�
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*(
_class
loc:@AssignMovingAvg/34720*
_output_shapes
:2
AssignMovingAvg/mul�
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_34720AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*(
_class
loc:@AssignMovingAvg/34720*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp�
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:GPU:0**
_class 
loc:@AssignMovingAvg_1/34726*
_output_shapes
: *
dtype0*
valueB
 *
�#<2
AssignMovingAvg_1/decay�
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_34726*
_output_shapes
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOp�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0**
_class 
loc:@AssignMovingAvg_1/34726*
_output_shapes
:2
AssignMovingAvg_1/sub�
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0**
_class 
loc:@AssignMovingAvg_1/34726*
_output_shapes
:2
AssignMovingAvg_1/mul�
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_34726AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0**
_class 
loc:@AssignMovingAvg_1/34726*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:2
batchnorm/add/y�
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/Rsqrt�
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/mul/ReadVariableOp�
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mul�
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*N
_output_shapes<
::8������������������������������������2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2�
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp�
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/sub�
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*N
_output_shapes<
::8������������������������������������2
batchnorm/add_1�
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*N
_output_shapes<
::8������������������������������������2

Identity"
identityIdentity:output:0*]
_input_shapesL
J:8������������������������������������::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:v r
N
_output_shapes<
::8������������������������������������
 
_user_specified_nameinputs
�

�
C__inference_conv3d_3_layer_call_and_return_conditional_losses_35180

inputs"
conv3d_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�Conv3D/ReadVariableOp�
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
: @*
dtype02
Conv3D/ReadVariableOp�
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������@*
paddingVALID*
strides	
2
Conv3D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������@2	
BiasAddd
ReluReluBiasAdd:output:0*
T0*3
_output_shapes!
:���������@2
Relu�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*
T0*3
_output_shapes!
:���������@2

Identity"
identityIdentity:output:0*:
_input_shapes)
':���������% ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:[ W
3
_output_shapes!
:���������% 
 
_user_specified_nameinputs
�
^
B__inference_flatten_layer_call_and_return_conditional_losses_35202

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"������  2
Consti
ReshapeReshapeinputsConst:output:0*
T0*)
_output_shapes
:�����������2	
Reshapef
IdentityIdentityReshape:output:0*
T0*)
_output_shapes
:�����������2

Identity"
identityIdentity:output:0*2
_input_shapes!
:���������@:[ W
3
_output_shapes!
:���������@
 
_user_specified_nameinputs
�
�
+__inference_normalize_2_layer_call_fn_36080

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������% *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*1J 8� *O
fJRH
F__inference_normalize_2_layer_call_and_return_conditional_losses_351132
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*3
_output_shapes!
:���������% 2

Identity"
identityIdentity:output:0*B
_input_shapes1
/:���������% ::::22
StatefulPartitionedCallStatefulPartitionedCall:[ W
3
_output_shapes!
:���������% 
 
_user_specified_nameinputs
�
}
(__inference_conv3d_2_layer_call_fn_35929

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������% *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*1J 8� *L
fGRE
C__inference_conv3d_2_layer_call_and_return_conditional_losses_350622
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*3
_output_shapes!
:���������% 2

Identity"
identityIdentity:output:0*:
_input_shapes)
':���������K::22
StatefulPartitionedCallStatefulPartitionedCall:[ W
3
_output_shapes!
:���������K
 
_user_specified_nameinputs
�
}
(__inference_conv3d_3_layer_call_fn_36113

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*1J 8� *L
fGRE
C__inference_conv3d_3_layer_call_and_return_conditional_losses_351802
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*3
_output_shapes!
:���������@2

Identity"
identityIdentity:output:0*:
_input_shapes)
':���������% ::22
StatefulPartitionedCallStatefulPartitionedCall:[ W
3
_output_shapes!
:���������% 
 
_user_specified_nameinputs
�
�
+__inference_normalize_1_layer_call_fn_35827

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *N
_output_shapes<
::8������������������������������������*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*1J 8� *O
fJRH
F__inference_normalize_1_layer_call_and_return_conditional_losses_347782
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*N
_output_shapes<
::8������������������������������������2

Identity"
identityIdentity:output:0*]
_input_shapesL
J:8������������������������������������::::22
StatefulPartitionedCallStatefulPartitionedCall:v r
N
_output_shapes<
::8������������������������������������
 
_user_specified_nameinputs
�
C
'__inference_flatten_layer_call_fn_36124

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:�����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*1J 8� *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_352022
PartitionedCalln
IdentityIdentityPartitionedCall:output:0*
T0*)
_output_shapes
:�����������2

Identity"
identityIdentity:output:0*2
_input_shapes!
:���������@:[ W
3
_output_shapes!
:���������@
 
_user_specified_nameinputs
�
�
+__inference_normalize_1_layer_call_fn_35814

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *N
_output_shapes<
::8������������������������������������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*1J 8� *O
fJRH
F__inference_normalize_1_layer_call_and_return_conditional_losses_347452
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*N
_output_shapes<
::8������������������������������������2

Identity"
identityIdentity:output:0*]
_input_shapesL
J:8������������������������������������::::22
StatefulPartitionedCallStatefulPartitionedCall:v r
N
_output_shapes<
::8������������������������������������
 
_user_specified_nameinputs
�

�
C__inference_conv3d_3_layer_call_and_return_conditional_losses_36104

inputs"
conv3d_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�Conv3D/ReadVariableOp�
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
: @*
dtype02
Conv3D/ReadVariableOp�
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������@*
paddingVALID*
strides	
2
Conv3D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������@2	
BiasAddd
ReluReluBiasAdd:output:0*
T0*3
_output_shapes!
:���������@2
Relu�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*
T0*3
_output_shapes!
:���������@2

Identity"
identityIdentity:output:0*:
_input_shapes)
':���������% ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:[ W
3
_output_shapes!
:���������% 
 
_user_specified_nameinputs
�%
�
B__inference_model_6_layer_call_and_return_conditional_losses_35238
input_7
conv3d_1_34955
conv3d_1_34957
normalize_1_35042
normalize_1_35044
normalize_1_35046
normalize_1_35048
conv3d_2_35073
conv3d_2_35075
normalize_2_35160
normalize_2_35162
normalize_2_35164
normalize_2_35166
conv3d_3_35191
conv3d_3_35193
theta_35232
theta_35234
identity�� conv3d_1/StatefulPartitionedCall� conv3d_2/StatefulPartitionedCall� conv3d_3/StatefulPartitionedCall�#normalize_1/StatefulPartitionedCall�#normalize_2/StatefulPartitionedCall�theta/StatefulPartitionedCall�
 conv3d_1/StatefulPartitionedCallStatefulPartitionedCallinput_7conv3d_1_34955conv3d_1_34957*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������K*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*1J 8� *L
fGRE
C__inference_conv3d_1_layer_call_and_return_conditional_losses_349442"
 conv3d_1/StatefulPartitionedCall�
#normalize_1/StatefulPartitionedCallStatefulPartitionedCall)conv3d_1/StatefulPartitionedCall:output:0normalize_1_35042normalize_1_35044normalize_1_35046normalize_1_35048*
Tin	
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������K*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*1J 8� *O
fJRH
F__inference_normalize_1_layer_call_and_return_conditional_losses_349952%
#normalize_1/StatefulPartitionedCall�
 conv3d_2/StatefulPartitionedCallStatefulPartitionedCall,normalize_1/StatefulPartitionedCall:output:0conv3d_2_35073conv3d_2_35075*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������% *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*1J 8� *L
fGRE
C__inference_conv3d_2_layer_call_and_return_conditional_losses_350622"
 conv3d_2/StatefulPartitionedCall�
#normalize_2/StatefulPartitionedCallStatefulPartitionedCall)conv3d_2/StatefulPartitionedCall:output:0normalize_2_35160normalize_2_35162normalize_2_35164normalize_2_35166*
Tin	
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������% *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*1J 8� *O
fJRH
F__inference_normalize_2_layer_call_and_return_conditional_losses_351132%
#normalize_2/StatefulPartitionedCall�
 conv3d_3/StatefulPartitionedCallStatefulPartitionedCall,normalize_2/StatefulPartitionedCall:output:0conv3d_3_35191conv3d_3_35193*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*1J 8� *L
fGRE
C__inference_conv3d_3_layer_call_and_return_conditional_losses_351802"
 conv3d_3/StatefulPartitionedCall�
flatten/PartitionedCallPartitionedCall)conv3d_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:�����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*1J 8� *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_352022
flatten/PartitionedCall�
theta/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0theta_35232theta_35234*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*1J 8� *I
fDRB
@__inference_theta_layer_call_and_return_conditional_losses_352212
theta/StatefulPartitionedCall�
IdentityIdentity&theta/StatefulPartitionedCall:output:0!^conv3d_1/StatefulPartitionedCall!^conv3d_2/StatefulPartitionedCall!^conv3d_3/StatefulPartitionedCall$^normalize_1/StatefulPartitionedCall$^normalize_2/StatefulPartitionedCall^theta/StatefulPartitionedCall*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*s
_input_shapesb
`:���������==�::::::::::::::::2D
 conv3d_1/StatefulPartitionedCall conv3d_1/StatefulPartitionedCall2D
 conv3d_2/StatefulPartitionedCall conv3d_2/StatefulPartitionedCall2D
 conv3d_3/StatefulPartitionedCall conv3d_3/StatefulPartitionedCall2J
#normalize_1/StatefulPartitionedCall#normalize_1/StatefulPartitionedCall2J
#normalize_2/StatefulPartitionedCall#normalize_2/StatefulPartitionedCall2>
theta/StatefulPartitionedCalltheta/StatefulPartitionedCall:] Y
4
_output_shapes"
 :���������==�
!
_user_specified_name	input_7
�
�
+__inference_normalize_2_layer_call_fn_35998

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *N
_output_shapes<
::8������������������������������������ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*1J 8� *O
fJRH
F__inference_normalize_2_layer_call_and_return_conditional_losses_348852
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*N
_output_shapes<
::8������������������������������������ 2

Identity"
identityIdentity:output:0*]
_input_shapesL
J:8������������������������������������ ::::22
StatefulPartitionedCallStatefulPartitionedCall:v r
N
_output_shapes<
::8������������������������������������ 
 
_user_specified_nameinputs
�
�
+__inference_normalize_1_layer_call_fn_35896

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������K*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*1J 8� *O
fJRH
F__inference_normalize_1_layer_call_and_return_conditional_losses_349952
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*3
_output_shapes!
:���������K2

Identity"
identityIdentity:output:0*B
_input_shapes1
/:���������K::::22
StatefulPartitionedCallStatefulPartitionedCall:[ W
3
_output_shapes!
:���������K
 
_user_specified_nameinputs
�

�
C__inference_conv3d_1_layer_call_and_return_conditional_losses_34944

inputs"
conv3d_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�Conv3D/ReadVariableOp�
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:*
dtype02
Conv3D/ReadVariableOp�
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������K*
paddingVALID*
strides	
2
Conv3D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������K2	
BiasAddd
ReluReluBiasAdd:output:0*
T0*3
_output_shapes!
:���������K2
Relu�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv3D/ReadVariableOp*
T0*3
_output_shapes!
:���������K2

Identity"
identityIdentity:output:0*;
_input_shapes*
(:���������==�::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv3D/ReadVariableOpConv3D/ReadVariableOp:\ X
4
_output_shapes"
 :���������==�
 
_user_specified_nameinputs
�2
�
F__inference_normalize_2_layer_call_and_return_conditional_losses_34885

inputs
assignmovingavg_34860
assignmovingavg_1_34866)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identity��#AssignMovingAvg/AssignSubVariableOp�AssignMovingAvg/ReadVariableOp�%AssignMovingAvg_1/AssignSubVariableOp� AssignMovingAvg_1/ReadVariableOp�batchnorm/ReadVariableOp�batchnorm/mul/ReadVariableOp�
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*%
valueB"             2 
moments/mean/reduction_indices�
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0**
_output_shapes
: *
	keep_dims(2
moments/mean�
moments/StopGradientStopGradientmoments/mean:output:0*
T0**
_output_shapes
: 2
moments/StopGradient�
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*N
_output_shapes<
::8������������������������������������ 2
moments/SquaredDifference�
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"moments/variance/reduction_indices�
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0**
_output_shapes
: *
	keep_dims(2
moments/variance�
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
: *
squeeze_dims
 2
moments/Squeeze�
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
: *
squeeze_dims
 2
moments/Squeeze_1�
AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:GPU:0*(
_class
loc:@AssignMovingAvg/34860*
_output_shapes
: *
dtype0*
valueB
 *
�#<2
AssignMovingAvg/decay�
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_34860*
_output_shapes
: *
dtype02 
AssignMovingAvg/ReadVariableOp�
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*(
_class
loc:@AssignMovingAvg/34860*
_output_shapes
: 2
AssignMovingAvg/sub�
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*(
_class
loc:@AssignMovingAvg/34860*
_output_shapes
: 2
AssignMovingAvg/mul�
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_34860AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*(
_class
loc:@AssignMovingAvg/34860*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp�
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:GPU:0**
_class 
loc:@AssignMovingAvg_1/34866*
_output_shapes
: *
dtype0*
valueB
 *
�#<2
AssignMovingAvg_1/decay�
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_34866*
_output_shapes
: *
dtype02"
 AssignMovingAvg_1/ReadVariableOp�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0**
_class 
loc:@AssignMovingAvg_1/34866*
_output_shapes
: 2
AssignMovingAvg_1/sub�
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0**
_class 
loc:@AssignMovingAvg_1/34866*
_output_shapes
: 2
AssignMovingAvg_1/mul�
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_34866AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0**
_class 
loc:@AssignMovingAvg_1/34866*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:2
batchnorm/add/y�
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
: 2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
: 2
batchnorm/Rsqrt�
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/mul/ReadVariableOp�
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 2
batchnorm/mul�
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*N
_output_shapes<
::8������������������������������������ 2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
: 2
batchnorm/mul_2�
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOp�
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
: 2
batchnorm/sub�
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*N
_output_shapes<
::8������������������������������������ 2
batchnorm/add_1�
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*N
_output_shapes<
::8������������������������������������ 2

Identity"
identityIdentity:output:0*]
_input_shapesL
J:8������������������������������������ ::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:v r
N
_output_shapes<
::8������������������������������������ 
 
_user_specified_nameinputs
�2
�
F__inference_normalize_1_layer_call_and_return_conditional_losses_35781

inputs
assignmovingavg_35756
assignmovingavg_1_35762)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identity��#AssignMovingAvg/AssignSubVariableOp�AssignMovingAvg/ReadVariableOp�%AssignMovingAvg_1/AssignSubVariableOp� AssignMovingAvg_1/ReadVariableOp�batchnorm/ReadVariableOp�batchnorm/mul/ReadVariableOp�
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*%
valueB"             2 
moments/mean/reduction_indices�
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0**
_output_shapes
:*
	keep_dims(2
moments/mean�
moments/StopGradientStopGradientmoments/mean:output:0*
T0**
_output_shapes
:2
moments/StopGradient�
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*N
_output_shapes<
::8������������������������������������2
moments/SquaredDifference�
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"moments/variance/reduction_indices�
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0**
_output_shapes
:*
	keep_dims(2
moments/variance�
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze�
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze_1�
AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:GPU:0*(
_class
loc:@AssignMovingAvg/35756*
_output_shapes
: *
dtype0*
valueB
 *
�#<2
AssignMovingAvg/decay�
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_35756*
_output_shapes
:*
dtype02 
AssignMovingAvg/ReadVariableOp�
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*(
_class
loc:@AssignMovingAvg/35756*
_output_shapes
:2
AssignMovingAvg/sub�
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*(
_class
loc:@AssignMovingAvg/35756*
_output_shapes
:2
AssignMovingAvg/mul�
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_35756AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*(
_class
loc:@AssignMovingAvg/35756*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp�
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:GPU:0**
_class 
loc:@AssignMovingAvg_1/35762*
_output_shapes
: *
dtype0*
valueB
 *
�#<2
AssignMovingAvg_1/decay�
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_35762*
_output_shapes
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOp�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0**
_class 
loc:@AssignMovingAvg_1/35762*
_output_shapes
:2
AssignMovingAvg_1/sub�
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0**
_class 
loc:@AssignMovingAvg_1/35762*
_output_shapes
:2
AssignMovingAvg_1/mul�
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_35762AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0**
_class 
loc:@AssignMovingAvg_1/35762*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:2
batchnorm/add/y�
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/Rsqrt�
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/mul/ReadVariableOp�
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mul�
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*N
_output_shapes<
::8������������������������������������2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2�
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp�
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/sub�
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*N
_output_shapes<
::8������������������������������������2
batchnorm/add_1�
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*N
_output_shapes<
::8������������������������������������2

Identity"
identityIdentity:output:0*]
_input_shapesL
J:8������������������������������������::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:v r
N
_output_shapes<
::8������������������������������������
 
_user_specified_nameinputs
�1
�
F__inference_normalize_2_layer_call_and_return_conditional_losses_36047

inputs
assignmovingavg_36022
assignmovingavg_1_36028)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identity��#AssignMovingAvg/AssignSubVariableOp�AssignMovingAvg/ReadVariableOp�%AssignMovingAvg_1/AssignSubVariableOp� AssignMovingAvg_1/ReadVariableOp�batchnorm/ReadVariableOp�batchnorm/mul/ReadVariableOp�
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*%
valueB"             2 
moments/mean/reduction_indices�
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0**
_output_shapes
: *
	keep_dims(2
moments/mean�
moments/StopGradientStopGradientmoments/mean:output:0*
T0**
_output_shapes
: 2
moments/StopGradient�
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*3
_output_shapes!
:���������% 2
moments/SquaredDifference�
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"moments/variance/reduction_indices�
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0**
_output_shapes
: *
	keep_dims(2
moments/variance�
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
: *
squeeze_dims
 2
moments/Squeeze�
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
: *
squeeze_dims
 2
moments/Squeeze_1�
AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:GPU:0*(
_class
loc:@AssignMovingAvg/36022*
_output_shapes
: *
dtype0*
valueB
 *
�#<2
AssignMovingAvg/decay�
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_36022*
_output_shapes
: *
dtype02 
AssignMovingAvg/ReadVariableOp�
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*(
_class
loc:@AssignMovingAvg/36022*
_output_shapes
: 2
AssignMovingAvg/sub�
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*(
_class
loc:@AssignMovingAvg/36022*
_output_shapes
: 2
AssignMovingAvg/mul�
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_36022AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*(
_class
loc:@AssignMovingAvg/36022*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp�
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:GPU:0**
_class 
loc:@AssignMovingAvg_1/36028*
_output_shapes
: *
dtype0*
valueB
 *
�#<2
AssignMovingAvg_1/decay�
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_36028*
_output_shapes
: *
dtype02"
 AssignMovingAvg_1/ReadVariableOp�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0**
_class 
loc:@AssignMovingAvg_1/36028*
_output_shapes
: 2
AssignMovingAvg_1/sub�
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0**
_class 
loc:@AssignMovingAvg_1/36028*
_output_shapes
: 2
AssignMovingAvg_1/mul�
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_36028AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0**
_class 
loc:@AssignMovingAvg_1/36028*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:2
batchnorm/add/y�
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
: 2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
: 2
batchnorm/Rsqrt�
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/mul/ReadVariableOp�
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 2
batchnorm/mul�
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*3
_output_shapes!
:���������% 2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
: 2
batchnorm/mul_2�
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOp�
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
: 2
batchnorm/sub�
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*3
_output_shapes!
:���������% 2
batchnorm/add_1�
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*3
_output_shapes!
:���������% 2

Identity"
identityIdentity:output:0*B
_input_shapes1
/:���������% ::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:[ W
3
_output_shapes!
:���������% 
 
_user_specified_nameinputs
�
�
F__inference_normalize_1_layer_call_and_return_conditional_losses_35883

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identity��batchnorm/ReadVariableOp�batchnorm/ReadVariableOp_1�batchnorm/ReadVariableOp_2�batchnorm/mul/ReadVariableOp�
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:2
batchnorm/add/y�
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/Rsqrt�
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/mul/ReadVariableOp�
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mul�
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*3
_output_shapes!
:���������K2
batchnorm/mul_1�
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp_1�
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2�
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp_2�
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/sub�
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*3
_output_shapes!
:���������K2
batchnorm/add_1�
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*3
_output_shapes!
:���������K2

Identity"
identityIdentity:output:0*B
_input_shapes1
/:���������K::::24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:[ W
3
_output_shapes!
:���������K
 
_user_specified_nameinputs
�
}
(__inference_conv3d_1_layer_call_fn_35745

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������K*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*1J 8� *L
fGRE
C__inference_conv3d_1_layer_call_and_return_conditional_losses_349442
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*3
_output_shapes!
:���������K2

Identity"
identityIdentity:output:0*;
_input_shapes*
(:���������==�::22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :���������==�
 
_user_specified_nameinputs
�
z
%__inference_theta_layer_call_fn_36144

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*1J 8� *I
fDRB
@__inference_theta_layer_call_and_return_conditional_losses_352212
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*0
_input_shapes
:�����������::22
StatefulPartitionedCallStatefulPartitionedCall:Q M
)
_output_shapes
:�����������
 
_user_specified_nameinputs
�	
�
@__inference_theta_layer_call_and_return_conditional_losses_35221

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*!
_output_shapes
:���*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2	
BiasAddb
SoftmaxSoftmaxBiasAdd:output:0*
T0*(
_output_shapes
:����������2	
Softmax�
IdentityIdentitySoftmax:softmax:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*0
_input_shapes
:�����������::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:Q M
)
_output_shapes
:�����������
 
_user_specified_nameinputs
�
�
B__inference_model_6_layer_call_and_return_conditional_losses_35585

inputs+
'conv3d_1_conv3d_readvariableop_resource,
(conv3d_1_biasadd_readvariableop_resource%
!normalize_1_assignmovingavg_35505'
#normalize_1_assignmovingavg_1_355115
1normalize_1_batchnorm_mul_readvariableop_resource1
-normalize_1_batchnorm_readvariableop_resource+
'conv3d_2_conv3d_readvariableop_resource,
(conv3d_2_biasadd_readvariableop_resource%
!normalize_2_assignmovingavg_35544'
#normalize_2_assignmovingavg_1_355505
1normalize_2_batchnorm_mul_readvariableop_resource1
-normalize_2_batchnorm_readvariableop_resource+
'conv3d_3_conv3d_readvariableop_resource,
(conv3d_3_biasadd_readvariableop_resource(
$theta_matmul_readvariableop_resource)
%theta_biasadd_readvariableop_resource
identity��conv3d_1/BiasAdd/ReadVariableOp�conv3d_1/Conv3D/ReadVariableOp�conv3d_2/BiasAdd/ReadVariableOp�conv3d_2/Conv3D/ReadVariableOp�conv3d_3/BiasAdd/ReadVariableOp�conv3d_3/Conv3D/ReadVariableOp�/normalize_1/AssignMovingAvg/AssignSubVariableOp�*normalize_1/AssignMovingAvg/ReadVariableOp�1normalize_1/AssignMovingAvg_1/AssignSubVariableOp�,normalize_1/AssignMovingAvg_1/ReadVariableOp�$normalize_1/batchnorm/ReadVariableOp�(normalize_1/batchnorm/mul/ReadVariableOp�/normalize_2/AssignMovingAvg/AssignSubVariableOp�*normalize_2/AssignMovingAvg/ReadVariableOp�1normalize_2/AssignMovingAvg_1/AssignSubVariableOp�,normalize_2/AssignMovingAvg_1/ReadVariableOp�$normalize_2/batchnorm/ReadVariableOp�(normalize_2/batchnorm/mul/ReadVariableOp�theta/BiasAdd/ReadVariableOp�theta/MatMul/ReadVariableOp�
conv3d_1/Conv3D/ReadVariableOpReadVariableOp'conv3d_1_conv3d_readvariableop_resource**
_output_shapes
:*
dtype02 
conv3d_1/Conv3D/ReadVariableOp�
conv3d_1/Conv3DConv3Dinputs&conv3d_1/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������K*
paddingVALID*
strides	
2
conv3d_1/Conv3D�
conv3d_1/BiasAdd/ReadVariableOpReadVariableOp(conv3d_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
conv3d_1/BiasAdd/ReadVariableOp�
conv3d_1/BiasAddBiasAddconv3d_1/Conv3D:output:0'conv3d_1/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������K2
conv3d_1/BiasAdd
conv3d_1/ReluReluconv3d_1/BiasAdd:output:0*
T0*3
_output_shapes!
:���������K2
conv3d_1/Relu�
*normalize_1/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*%
valueB"             2,
*normalize_1/moments/mean/reduction_indices�
normalize_1/moments/meanMeanconv3d_1/Relu:activations:03normalize_1/moments/mean/reduction_indices:output:0*
T0**
_output_shapes
:*
	keep_dims(2
normalize_1/moments/mean�
 normalize_1/moments/StopGradientStopGradient!normalize_1/moments/mean:output:0*
T0**
_output_shapes
:2"
 normalize_1/moments/StopGradient�
%normalize_1/moments/SquaredDifferenceSquaredDifferenceconv3d_1/Relu:activations:0)normalize_1/moments/StopGradient:output:0*
T0*3
_output_shapes!
:���������K2'
%normalize_1/moments/SquaredDifference�
.normalize_1/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*%
valueB"             20
.normalize_1/moments/variance/reduction_indices�
normalize_1/moments/varianceMean)normalize_1/moments/SquaredDifference:z:07normalize_1/moments/variance/reduction_indices:output:0*
T0**
_output_shapes
:*
	keep_dims(2
normalize_1/moments/variance�
normalize_1/moments/SqueezeSqueeze!normalize_1/moments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
normalize_1/moments/Squeeze�
normalize_1/moments/Squeeze_1Squeeze%normalize_1/moments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
normalize_1/moments/Squeeze_1�
!normalize_1/AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:GPU:0*4
_class*
(&loc:@normalize_1/AssignMovingAvg/35505*
_output_shapes
: *
dtype0*
valueB
 *
�#<2#
!normalize_1/AssignMovingAvg/decay�
*normalize_1/AssignMovingAvg/ReadVariableOpReadVariableOp!normalize_1_assignmovingavg_35505*
_output_shapes
:*
dtype02,
*normalize_1/AssignMovingAvg/ReadVariableOp�
normalize_1/AssignMovingAvg/subSub2normalize_1/AssignMovingAvg/ReadVariableOp:value:0$normalize_1/moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*4
_class*
(&loc:@normalize_1/AssignMovingAvg/35505*
_output_shapes
:2!
normalize_1/AssignMovingAvg/sub�
normalize_1/AssignMovingAvg/mulMul#normalize_1/AssignMovingAvg/sub:z:0*normalize_1/AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*4
_class*
(&loc:@normalize_1/AssignMovingAvg/35505*
_output_shapes
:2!
normalize_1/AssignMovingAvg/mul�
/normalize_1/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp!normalize_1_assignmovingavg_35505#normalize_1/AssignMovingAvg/mul:z:0+^normalize_1/AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*4
_class*
(&loc:@normalize_1/AssignMovingAvg/35505*
_output_shapes
 *
dtype021
/normalize_1/AssignMovingAvg/AssignSubVariableOp�
#normalize_1/AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:GPU:0*6
_class,
*(loc:@normalize_1/AssignMovingAvg_1/35511*
_output_shapes
: *
dtype0*
valueB
 *
�#<2%
#normalize_1/AssignMovingAvg_1/decay�
,normalize_1/AssignMovingAvg_1/ReadVariableOpReadVariableOp#normalize_1_assignmovingavg_1_35511*
_output_shapes
:*
dtype02.
,normalize_1/AssignMovingAvg_1/ReadVariableOp�
!normalize_1/AssignMovingAvg_1/subSub4normalize_1/AssignMovingAvg_1/ReadVariableOp:value:0&normalize_1/moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*6
_class,
*(loc:@normalize_1/AssignMovingAvg_1/35511*
_output_shapes
:2#
!normalize_1/AssignMovingAvg_1/sub�
!normalize_1/AssignMovingAvg_1/mulMul%normalize_1/AssignMovingAvg_1/sub:z:0,normalize_1/AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*6
_class,
*(loc:@normalize_1/AssignMovingAvg_1/35511*
_output_shapes
:2#
!normalize_1/AssignMovingAvg_1/mul�
1normalize_1/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp#normalize_1_assignmovingavg_1_35511%normalize_1/AssignMovingAvg_1/mul:z:0-^normalize_1/AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*6
_class,
*(loc:@normalize_1/AssignMovingAvg_1/35511*
_output_shapes
 *
dtype023
1normalize_1/AssignMovingAvg_1/AssignSubVariableOp
normalize_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:2
normalize_1/batchnorm/add/y�
normalize_1/batchnorm/addAddV2&normalize_1/moments/Squeeze_1:output:0$normalize_1/batchnorm/add/y:output:0*
T0*
_output_shapes
:2
normalize_1/batchnorm/add�
normalize_1/batchnorm/RsqrtRsqrtnormalize_1/batchnorm/add:z:0*
T0*
_output_shapes
:2
normalize_1/batchnorm/Rsqrt�
(normalize_1/batchnorm/mul/ReadVariableOpReadVariableOp1normalize_1_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02*
(normalize_1/batchnorm/mul/ReadVariableOp�
normalize_1/batchnorm/mulMulnormalize_1/batchnorm/Rsqrt:y:00normalize_1/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2
normalize_1/batchnorm/mul�
normalize_1/batchnorm/mul_1Mulconv3d_1/Relu:activations:0normalize_1/batchnorm/mul:z:0*
T0*3
_output_shapes!
:���������K2
normalize_1/batchnorm/mul_1�
normalize_1/batchnorm/mul_2Mul$normalize_1/moments/Squeeze:output:0normalize_1/batchnorm/mul:z:0*
T0*
_output_shapes
:2
normalize_1/batchnorm/mul_2�
$normalize_1/batchnorm/ReadVariableOpReadVariableOp-normalize_1_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02&
$normalize_1/batchnorm/ReadVariableOp�
normalize_1/batchnorm/subSub,normalize_1/batchnorm/ReadVariableOp:value:0normalize_1/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
normalize_1/batchnorm/sub�
normalize_1/batchnorm/add_1AddV2normalize_1/batchnorm/mul_1:z:0normalize_1/batchnorm/sub:z:0*
T0*3
_output_shapes!
:���������K2
normalize_1/batchnorm/add_1�
conv3d_2/Conv3D/ReadVariableOpReadVariableOp'conv3d_2_conv3d_readvariableop_resource**
_output_shapes
: *
dtype02 
conv3d_2/Conv3D/ReadVariableOp�
conv3d_2/Conv3DConv3Dnormalize_1/batchnorm/add_1:z:0&conv3d_2/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������% *
paddingVALID*
strides	
2
conv3d_2/Conv3D�
conv3d_2/BiasAdd/ReadVariableOpReadVariableOp(conv3d_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
conv3d_2/BiasAdd/ReadVariableOp�
conv3d_2/BiasAddBiasAddconv3d_2/Conv3D:output:0'conv3d_2/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������% 2
conv3d_2/BiasAdd
conv3d_2/ReluReluconv3d_2/BiasAdd:output:0*
T0*3
_output_shapes!
:���������% 2
conv3d_2/Relu�
*normalize_2/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*%
valueB"             2,
*normalize_2/moments/mean/reduction_indices�
normalize_2/moments/meanMeanconv3d_2/Relu:activations:03normalize_2/moments/mean/reduction_indices:output:0*
T0**
_output_shapes
: *
	keep_dims(2
normalize_2/moments/mean�
 normalize_2/moments/StopGradientStopGradient!normalize_2/moments/mean:output:0*
T0**
_output_shapes
: 2"
 normalize_2/moments/StopGradient�
%normalize_2/moments/SquaredDifferenceSquaredDifferenceconv3d_2/Relu:activations:0)normalize_2/moments/StopGradient:output:0*
T0*3
_output_shapes!
:���������% 2'
%normalize_2/moments/SquaredDifference�
.normalize_2/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*%
valueB"             20
.normalize_2/moments/variance/reduction_indices�
normalize_2/moments/varianceMean)normalize_2/moments/SquaredDifference:z:07normalize_2/moments/variance/reduction_indices:output:0*
T0**
_output_shapes
: *
	keep_dims(2
normalize_2/moments/variance�
normalize_2/moments/SqueezeSqueeze!normalize_2/moments/mean:output:0*
T0*
_output_shapes
: *
squeeze_dims
 2
normalize_2/moments/Squeeze�
normalize_2/moments/Squeeze_1Squeeze%normalize_2/moments/variance:output:0*
T0*
_output_shapes
: *
squeeze_dims
 2
normalize_2/moments/Squeeze_1�
!normalize_2/AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:GPU:0*4
_class*
(&loc:@normalize_2/AssignMovingAvg/35544*
_output_shapes
: *
dtype0*
valueB
 *
�#<2#
!normalize_2/AssignMovingAvg/decay�
*normalize_2/AssignMovingAvg/ReadVariableOpReadVariableOp!normalize_2_assignmovingavg_35544*
_output_shapes
: *
dtype02,
*normalize_2/AssignMovingAvg/ReadVariableOp�
normalize_2/AssignMovingAvg/subSub2normalize_2/AssignMovingAvg/ReadVariableOp:value:0$normalize_2/moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*4
_class*
(&loc:@normalize_2/AssignMovingAvg/35544*
_output_shapes
: 2!
normalize_2/AssignMovingAvg/sub�
normalize_2/AssignMovingAvg/mulMul#normalize_2/AssignMovingAvg/sub:z:0*normalize_2/AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*4
_class*
(&loc:@normalize_2/AssignMovingAvg/35544*
_output_shapes
: 2!
normalize_2/AssignMovingAvg/mul�
/normalize_2/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp!normalize_2_assignmovingavg_35544#normalize_2/AssignMovingAvg/mul:z:0+^normalize_2/AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*4
_class*
(&loc:@normalize_2/AssignMovingAvg/35544*
_output_shapes
 *
dtype021
/normalize_2/AssignMovingAvg/AssignSubVariableOp�
#normalize_2/AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:GPU:0*6
_class,
*(loc:@normalize_2/AssignMovingAvg_1/35550*
_output_shapes
: *
dtype0*
valueB
 *
�#<2%
#normalize_2/AssignMovingAvg_1/decay�
,normalize_2/AssignMovingAvg_1/ReadVariableOpReadVariableOp#normalize_2_assignmovingavg_1_35550*
_output_shapes
: *
dtype02.
,normalize_2/AssignMovingAvg_1/ReadVariableOp�
!normalize_2/AssignMovingAvg_1/subSub4normalize_2/AssignMovingAvg_1/ReadVariableOp:value:0&normalize_2/moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*6
_class,
*(loc:@normalize_2/AssignMovingAvg_1/35550*
_output_shapes
: 2#
!normalize_2/AssignMovingAvg_1/sub�
!normalize_2/AssignMovingAvg_1/mulMul%normalize_2/AssignMovingAvg_1/sub:z:0,normalize_2/AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*6
_class,
*(loc:@normalize_2/AssignMovingAvg_1/35550*
_output_shapes
: 2#
!normalize_2/AssignMovingAvg_1/mul�
1normalize_2/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp#normalize_2_assignmovingavg_1_35550%normalize_2/AssignMovingAvg_1/mul:z:0-^normalize_2/AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*6
_class,
*(loc:@normalize_2/AssignMovingAvg_1/35550*
_output_shapes
 *
dtype023
1normalize_2/AssignMovingAvg_1/AssignSubVariableOp
normalize_2/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:2
normalize_2/batchnorm/add/y�
normalize_2/batchnorm/addAddV2&normalize_2/moments/Squeeze_1:output:0$normalize_2/batchnorm/add/y:output:0*
T0*
_output_shapes
: 2
normalize_2/batchnorm/add�
normalize_2/batchnorm/RsqrtRsqrtnormalize_2/batchnorm/add:z:0*
T0*
_output_shapes
: 2
normalize_2/batchnorm/Rsqrt�
(normalize_2/batchnorm/mul/ReadVariableOpReadVariableOp1normalize_2_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02*
(normalize_2/batchnorm/mul/ReadVariableOp�
normalize_2/batchnorm/mulMulnormalize_2/batchnorm/Rsqrt:y:00normalize_2/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 2
normalize_2/batchnorm/mul�
normalize_2/batchnorm/mul_1Mulconv3d_2/Relu:activations:0normalize_2/batchnorm/mul:z:0*
T0*3
_output_shapes!
:���������% 2
normalize_2/batchnorm/mul_1�
normalize_2/batchnorm/mul_2Mul$normalize_2/moments/Squeeze:output:0normalize_2/batchnorm/mul:z:0*
T0*
_output_shapes
: 2
normalize_2/batchnorm/mul_2�
$normalize_2/batchnorm/ReadVariableOpReadVariableOp-normalize_2_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype02&
$normalize_2/batchnorm/ReadVariableOp�
normalize_2/batchnorm/subSub,normalize_2/batchnorm/ReadVariableOp:value:0normalize_2/batchnorm/mul_2:z:0*
T0*
_output_shapes
: 2
normalize_2/batchnorm/sub�
normalize_2/batchnorm/add_1AddV2normalize_2/batchnorm/mul_1:z:0normalize_2/batchnorm/sub:z:0*
T0*3
_output_shapes!
:���������% 2
normalize_2/batchnorm/add_1�
conv3d_3/Conv3D/ReadVariableOpReadVariableOp'conv3d_3_conv3d_readvariableop_resource**
_output_shapes
: @*
dtype02 
conv3d_3/Conv3D/ReadVariableOp�
conv3d_3/Conv3DConv3Dnormalize_2/batchnorm/add_1:z:0&conv3d_3/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������@*
paddingVALID*
strides	
2
conv3d_3/Conv3D�
conv3d_3/BiasAdd/ReadVariableOpReadVariableOp(conv3d_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv3d_3/BiasAdd/ReadVariableOp�
conv3d_3/BiasAddBiasAddconv3d_3/Conv3D:output:0'conv3d_3/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:���������@2
conv3d_3/BiasAdd
conv3d_3/ReluReluconv3d_3/BiasAdd:output:0*
T0*3
_output_shapes!
:���������@2
conv3d_3/Reluo
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"������  2
flatten/Const�
flatten/ReshapeReshapeconv3d_3/Relu:activations:0flatten/Const:output:0*
T0*)
_output_shapes
:�����������2
flatten/Reshape�
theta/MatMul/ReadVariableOpReadVariableOp$theta_matmul_readvariableop_resource*!
_output_shapes
:���*
dtype02
theta/MatMul/ReadVariableOp�
theta/MatMulMatMulflatten/Reshape:output:0#theta/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
theta/MatMul�
theta/BiasAdd/ReadVariableOpReadVariableOp%theta_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
theta/BiasAdd/ReadVariableOp�
theta/BiasAddBiasAddtheta/MatMul:product:0$theta/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
theta/BiasAddt
theta/SoftmaxSoftmaxtheta/BiasAdd:output:0*
T0*(
_output_shapes
:����������2
theta/Softmax�
IdentityIdentitytheta/Softmax:softmax:0 ^conv3d_1/BiasAdd/ReadVariableOp^conv3d_1/Conv3D/ReadVariableOp ^conv3d_2/BiasAdd/ReadVariableOp^conv3d_2/Conv3D/ReadVariableOp ^conv3d_3/BiasAdd/ReadVariableOp^conv3d_3/Conv3D/ReadVariableOp0^normalize_1/AssignMovingAvg/AssignSubVariableOp+^normalize_1/AssignMovingAvg/ReadVariableOp2^normalize_1/AssignMovingAvg_1/AssignSubVariableOp-^normalize_1/AssignMovingAvg_1/ReadVariableOp%^normalize_1/batchnorm/ReadVariableOp)^normalize_1/batchnorm/mul/ReadVariableOp0^normalize_2/AssignMovingAvg/AssignSubVariableOp+^normalize_2/AssignMovingAvg/ReadVariableOp2^normalize_2/AssignMovingAvg_1/AssignSubVariableOp-^normalize_2/AssignMovingAvg_1/ReadVariableOp%^normalize_2/batchnorm/ReadVariableOp)^normalize_2/batchnorm/mul/ReadVariableOp^theta/BiasAdd/ReadVariableOp^theta/MatMul/ReadVariableOp*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*s
_input_shapesb
`:���������==�::::::::::::::::2B
conv3d_1/BiasAdd/ReadVariableOpconv3d_1/BiasAdd/ReadVariableOp2@
conv3d_1/Conv3D/ReadVariableOpconv3d_1/Conv3D/ReadVariableOp2B
conv3d_2/BiasAdd/ReadVariableOpconv3d_2/BiasAdd/ReadVariableOp2@
conv3d_2/Conv3D/ReadVariableOpconv3d_2/Conv3D/ReadVariableOp2B
conv3d_3/BiasAdd/ReadVariableOpconv3d_3/BiasAdd/ReadVariableOp2@
conv3d_3/Conv3D/ReadVariableOpconv3d_3/Conv3D/ReadVariableOp2b
/normalize_1/AssignMovingAvg/AssignSubVariableOp/normalize_1/AssignMovingAvg/AssignSubVariableOp2X
*normalize_1/AssignMovingAvg/ReadVariableOp*normalize_1/AssignMovingAvg/ReadVariableOp2f
1normalize_1/AssignMovingAvg_1/AssignSubVariableOp1normalize_1/AssignMovingAvg_1/AssignSubVariableOp2\
,normalize_1/AssignMovingAvg_1/ReadVariableOp,normalize_1/AssignMovingAvg_1/ReadVariableOp2L
$normalize_1/batchnorm/ReadVariableOp$normalize_1/batchnorm/ReadVariableOp2T
(normalize_1/batchnorm/mul/ReadVariableOp(normalize_1/batchnorm/mul/ReadVariableOp2b
/normalize_2/AssignMovingAvg/AssignSubVariableOp/normalize_2/AssignMovingAvg/AssignSubVariableOp2X
*normalize_2/AssignMovingAvg/ReadVariableOp*normalize_2/AssignMovingAvg/ReadVariableOp2f
1normalize_2/AssignMovingAvg_1/AssignSubVariableOp1normalize_2/AssignMovingAvg_1/AssignSubVariableOp2\
,normalize_2/AssignMovingAvg_1/ReadVariableOp,normalize_2/AssignMovingAvg_1/ReadVariableOp2L
$normalize_2/batchnorm/ReadVariableOp$normalize_2/batchnorm/ReadVariableOp2T
(normalize_2/batchnorm/mul/ReadVariableOp(normalize_2/batchnorm/mul/ReadVariableOp2<
theta/BiasAdd/ReadVariableOptheta/BiasAdd/ReadVariableOp2:
theta/MatMul/ReadVariableOptheta/MatMul/ReadVariableOp:\ X
4
_output_shapes"
 :���������==�
 
_user_specified_nameinputs
�

�
'__inference_model_6_layer_call_fn_35362
input_7
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_7unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*.
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*1J 8� *K
fFRD
B__inference_model_6_layer_call_and_return_conditional_losses_353272
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*s
_input_shapesb
`:���������==�::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:] Y
4
_output_shapes"
 :���������==�
!
_user_specified_name	input_7
�1
�
F__inference_normalize_2_layer_call_and_return_conditional_losses_35113

inputs
assignmovingavg_35088
assignmovingavg_1_35094)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identity��#AssignMovingAvg/AssignSubVariableOp�AssignMovingAvg/ReadVariableOp�%AssignMovingAvg_1/AssignSubVariableOp� AssignMovingAvg_1/ReadVariableOp�batchnorm/ReadVariableOp�batchnorm/mul/ReadVariableOp�
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*%
valueB"             2 
moments/mean/reduction_indices�
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0**
_output_shapes
: *
	keep_dims(2
moments/mean�
moments/StopGradientStopGradientmoments/mean:output:0*
T0**
_output_shapes
: 2
moments/StopGradient�
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*3
_output_shapes!
:���������% 2
moments/SquaredDifference�
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"moments/variance/reduction_indices�
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0**
_output_shapes
: *
	keep_dims(2
moments/variance�
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
: *
squeeze_dims
 2
moments/Squeeze�
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
: *
squeeze_dims
 2
moments/Squeeze_1�
AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:GPU:0*(
_class
loc:@AssignMovingAvg/35088*
_output_shapes
: *
dtype0*
valueB
 *
�#<2
AssignMovingAvg/decay�
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_35088*
_output_shapes
: *
dtype02 
AssignMovingAvg/ReadVariableOp�
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*(
_class
loc:@AssignMovingAvg/35088*
_output_shapes
: 2
AssignMovingAvg/sub�
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*(
_class
loc:@AssignMovingAvg/35088*
_output_shapes
: 2
AssignMovingAvg/mul�
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_35088AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*(
_class
loc:@AssignMovingAvg/35088*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp�
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:GPU:0**
_class 
loc:@AssignMovingAvg_1/35094*
_output_shapes
: *
dtype0*
valueB
 *
�#<2
AssignMovingAvg_1/decay�
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_35094*
_output_shapes
: *
dtype02"
 AssignMovingAvg_1/ReadVariableOp�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0**
_class 
loc:@AssignMovingAvg_1/35094*
_output_shapes
: 2
AssignMovingAvg_1/sub�
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0**
_class 
loc:@AssignMovingAvg_1/35094*
_output_shapes
: 2
AssignMovingAvg_1/mul�
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_35094AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0**
_class 
loc:@AssignMovingAvg_1/35094*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:2
batchnorm/add/y�
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
: 2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
: 2
batchnorm/Rsqrt�
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/mul/ReadVariableOp�
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 2
batchnorm/mul�
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*3
_output_shapes!
:���������% 2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
: 2
batchnorm/mul_2�
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOp�
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
: 2
batchnorm/sub�
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*3
_output_shapes!
:���������% 2
batchnorm/add_1�
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*3
_output_shapes!
:���������% 2

Identity"
identityIdentity:output:0*B
_input_shapes1
/:���������% ::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:[ W
3
_output_shapes!
:���������% 
 
_user_specified_nameinputs"�L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
H
input_7=
serving_default_input_7:0���������==�:
theta1
StatefulPartitionedCall:0����������tensorflow/serving/predict:�
�R
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
layer_with_weights-4
layer-5
layer-6
layer_with_weights-5
layer-7
		optimizer

regularization_losses
trainable_variables
	variables
	keras_api

signatures
*n&call_and_return_all_conditional_losses
o__call__
p_default_save_signature"�N
_tf_keras_network�N{"class_name": "Functional", "name": "model_6", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "model_6", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 61, 61, 150, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_7"}, "name": "input_7", "inbound_nodes": []}, {"class_name": "Conv3D", "config": {"name": "conv3d_1", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [2, 2, 2]}, "strides": {"class_name": "__tuple__", "items": [2, 2, 2]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv3d_1", "inbound_nodes": [[["input_7", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "normalize_1", "trainable": true, "dtype": "float32", "axis": [4], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "normalize_1", "inbound_nodes": [[["conv3d_1", 0, 0, {}]]]}, {"class_name": "Conv3D", "config": {"name": "conv3d_2", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [2, 2, 2]}, "strides": {"class_name": "__tuple__", "items": [2, 2, 2]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv3d_2", "inbound_nodes": [[["normalize_1", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "normalize_2", "trainable": true, "dtype": "float32", "axis": [4], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "normalize_2", "inbound_nodes": [[["conv3d_2", 0, 0, {}]]]}, {"class_name": "Conv3D", "config": {"name": "conv3d_3", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [2, 2, 2]}, "strides": {"class_name": "__tuple__", "items": [2, 2, 2]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv3d_3", "inbound_nodes": [[["normalize_2", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten", "inbound_nodes": [[["conv3d_3", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "theta", "trainable": true, "dtype": "float32", "units": 360, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "theta", "inbound_nodes": [[["flatten", 0, 0, {}]]]}], "input_layers": [["input_7", 0, 0]], "output_layers": [["theta", 0, 0]]}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 61, 61, 150, 1]}, "ndim": 5, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 61, 61, 150, 1]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "model_6", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 61, 61, 150, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_7"}, "name": "input_7", "inbound_nodes": []}, {"class_name": "Conv3D", "config": {"name": "conv3d_1", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [2, 2, 2]}, "strides": {"class_name": "__tuple__", "items": [2, 2, 2]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv3d_1", "inbound_nodes": [[["input_7", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "normalize_1", "trainable": true, "dtype": "float32", "axis": [4], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "normalize_1", "inbound_nodes": [[["conv3d_1", 0, 0, {}]]]}, {"class_name": "Conv3D", "config": {"name": "conv3d_2", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [2, 2, 2]}, "strides": {"class_name": "__tuple__", "items": [2, 2, 2]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv3d_2", "inbound_nodes": [[["normalize_1", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "normalize_2", "trainable": true, "dtype": "float32", "axis": [4], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "normalize_2", "inbound_nodes": [[["conv3d_2", 0, 0, {}]]]}, {"class_name": "Conv3D", "config": {"name": "conv3d_3", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [2, 2, 2]}, "strides": {"class_name": "__tuple__", "items": [2, 2, 2]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv3d_3", "inbound_nodes": [[["normalize_2", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten", "inbound_nodes": [[["conv3d_3", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "theta", "trainable": true, "dtype": "float32", "units": 360, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "theta", "inbound_nodes": [[["flatten", 0, 0, {}]]]}], "input_layers": [["input_7", 0, 0]], "output_layers": [["theta", 0, 0]]}}, "training_config": {"loss": {"class_name": "BinaryCrossentropy", "config": {"reduction": "auto", "name": "binary_crossentropy", "from_logits": false, "label_smoothing": 0}}, "metrics": null, "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "SGD", "config": {"name": "SGD", "learning_rate": 0.20000000298023224, "decay": 0.0, "momentum": 0.0, "nesterov": false}}}}
�"�
_tf_keras_input_layer�{"class_name": "InputLayer", "name": "input_7", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 61, 61, 150, 1]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 61, 61, 150, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_7"}}
�	

kernel
bias
regularization_losses
trainable_variables
	variables
	keras_api
*q&call_and_return_all_conditional_losses
r__call__"�
_tf_keras_layer�{"class_name": "Conv3D", "name": "conv3d_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv3d_1", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [2, 2, 2]}, "strides": {"class_name": "__tuple__", "items": [2, 2, 2]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 5, "axes": {"-1": 1}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 61, 61, 150, 1]}}
�	
axis
	gamma
beta
moving_mean
moving_variance
regularization_losses
trainable_variables
	variables
	keras_api
*s&call_and_return_all_conditional_losses
t__call__"�
_tf_keras_layer�{"class_name": "BatchNormalization", "name": "normalize_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "normalize_1", "trainable": true, "dtype": "float32", "axis": [4], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 5, "max_ndim": null, "min_ndim": null, "axes": {"4": 16}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 30, 30, 75, 16]}}
�


kernel
bias
 regularization_losses
!trainable_variables
"	variables
#	keras_api
*u&call_and_return_all_conditional_losses
v__call__"�
_tf_keras_layer�{"class_name": "Conv3D", "name": "conv3d_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv3d_2", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [2, 2, 2]}, "strides": {"class_name": "__tuple__", "items": [2, 2, 2]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 5, "axes": {"-1": 16}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 30, 30, 75, 16]}}
�	
$axis
	%gamma
&beta
'moving_mean
(moving_variance
)regularization_losses
*trainable_variables
+	variables
,	keras_api
*w&call_and_return_all_conditional_losses
x__call__"�
_tf_keras_layer�{"class_name": "BatchNormalization", "name": "normalize_2", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "normalize_2", "trainable": true, "dtype": "float32", "axis": [4], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 5, "max_ndim": null, "min_ndim": null, "axes": {"4": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 15, 15, 37, 32]}}
�


-kernel
.bias
/regularization_losses
0trainable_variables
1	variables
2	keras_api
*y&call_and_return_all_conditional_losses
z__call__"�
_tf_keras_layer�{"class_name": "Conv3D", "name": "conv3d_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv3d_3", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [2, 2, 2]}, "strides": {"class_name": "__tuple__", "items": [2, 2, 2]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 5, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 15, 15, 37, 32]}}
�
3regularization_losses
4trainable_variables
5	variables
6	keras_api
*{&call_and_return_all_conditional_losses
|__call__"�
_tf_keras_layer�{"class_name": "Flatten", "name": "flatten", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
�

7kernel
8bias
9regularization_losses
:trainable_variables
;	variables
<	keras_api
*}&call_and_return_all_conditional_losses
~__call__"�
_tf_keras_layer�{"class_name": "Dense", "name": "theta", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "theta", "trainable": true, "dtype": "float32", "units": 360, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 56448}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 56448]}}
I
=iter
	>decay
?learning_rate
@momentum"
	optimizer
 "
trackable_list_wrapper
v
0
1
2
3
4
5
%6
&7
-8
.9
710
811"
trackable_list_wrapper
�
0
1
2
3
4
5
6
7
%8
&9
'10
(11
-12
.13
714
815"
trackable_list_wrapper
�
Alayer_metrics

Blayers
Cnon_trainable_variables
Dmetrics

regularization_losses
trainable_variables
Elayer_regularization_losses
	variables
o__call__
p_default_save_signature
*n&call_and_return_all_conditional_losses
&n"call_and_return_conditional_losses"
_generic_user_object
,
serving_default"
signature_map
-:+2conv3d_1/kernel
:2conv3d_1/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
�
Flayer_metrics

Glayers
Hmetrics
Inon_trainable_variables
regularization_losses
trainable_variables
Jlayer_regularization_losses
	variables
r__call__
*q&call_and_return_all_conditional_losses
&q"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
:2normalize_1/gamma
:2normalize_1/beta
':% (2normalize_1/moving_mean
+:) (2normalize_1/moving_variance
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
<
0
1
2
3"
trackable_list_wrapper
�
Klayer_metrics

Llayers
Mmetrics
Nnon_trainable_variables
regularization_losses
trainable_variables
Olayer_regularization_losses
	variables
t__call__
*s&call_and_return_all_conditional_losses
&s"call_and_return_conditional_losses"
_generic_user_object
-:+ 2conv3d_2/kernel
: 2conv3d_2/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
�
Player_metrics

Qlayers
Rmetrics
Snon_trainable_variables
 regularization_losses
!trainable_variables
Tlayer_regularization_losses
"	variables
v__call__
*u&call_and_return_all_conditional_losses
&u"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
: 2normalize_2/gamma
: 2normalize_2/beta
':%  (2normalize_2/moving_mean
+:)  (2normalize_2/moving_variance
 "
trackable_list_wrapper
.
%0
&1"
trackable_list_wrapper
<
%0
&1
'2
(3"
trackable_list_wrapper
�
Ulayer_metrics

Vlayers
Wmetrics
Xnon_trainable_variables
)regularization_losses
*trainable_variables
Ylayer_regularization_losses
+	variables
x__call__
*w&call_and_return_all_conditional_losses
&w"call_and_return_conditional_losses"
_generic_user_object
-:+ @2conv3d_3/kernel
:@2conv3d_3/bias
 "
trackable_list_wrapper
.
-0
.1"
trackable_list_wrapper
.
-0
.1"
trackable_list_wrapper
�
Zlayer_metrics

[layers
\metrics
]non_trainable_variables
/regularization_losses
0trainable_variables
^layer_regularization_losses
1	variables
z__call__
*y&call_and_return_all_conditional_losses
&y"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
_layer_metrics

`layers
ametrics
bnon_trainable_variables
3regularization_losses
4trainable_variables
clayer_regularization_losses
5	variables
|__call__
*{&call_and_return_all_conditional_losses
&{"call_and_return_conditional_losses"
_generic_user_object
!:���2theta/kernel
:�2
theta/bias
 "
trackable_list_wrapper
.
70
81"
trackable_list_wrapper
.
70
81"
trackable_list_wrapper
�
dlayer_metrics

elayers
fmetrics
gnon_trainable_variables
9regularization_losses
:trainable_variables
hlayer_regularization_losses
;	variables
~__call__
*}&call_and_return_all_conditional_losses
&}"call_and_return_conditional_losses"
_generic_user_object
:	 (2SGD/iter
: (2	SGD/decay
: (2SGD/learning_rate
: (2SGD/momentum
 "
trackable_dict_wrapper
X
0
1
2
3
4
5
6
7"
trackable_list_wrapper
<
0
1
'2
(3"
trackable_list_wrapper
'
i0"
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
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
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
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
'0
(1"
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
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
	jtotal
	kcount
l	variables
m	keras_api"�
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}
:  (2total
:  (2count
.
j0
k1"
trackable_list_wrapper
-
l	variables"
_generic_user_object
�2�
B__inference_model_6_layer_call_and_return_conditional_losses_35585
B__inference_model_6_layer_call_and_return_conditional_losses_35651
B__inference_model_6_layer_call_and_return_conditional_losses_35238
B__inference_model_6_layer_call_and_return_conditional_losses_35281�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
'__inference_model_6_layer_call_fn_35725
'__inference_model_6_layer_call_fn_35362
'__inference_model_6_layer_call_fn_35442
'__inference_model_6_layer_call_fn_35688�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
 __inference__wrapped_model_34649�
���
FullArgSpec
args� 
varargsjargs
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *3�0
.�+
input_7���������==�
�2�
C__inference_conv3d_1_layer_call_and_return_conditional_losses_35736�
���
FullArgSpec
args�
jself
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
annotations� *
 
�2�
(__inference_conv3d_1_layer_call_fn_35745�
���
FullArgSpec
args�
jself
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
annotations� *
 
�2�
F__inference_normalize_1_layer_call_and_return_conditional_losses_35801
F__inference_normalize_1_layer_call_and_return_conditional_losses_35781
F__inference_normalize_1_layer_call_and_return_conditional_losses_35863
F__inference_normalize_1_layer_call_and_return_conditional_losses_35883�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
+__inference_normalize_1_layer_call_fn_35814
+__inference_normalize_1_layer_call_fn_35827
+__inference_normalize_1_layer_call_fn_35909
+__inference_normalize_1_layer_call_fn_35896�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
C__inference_conv3d_2_layer_call_and_return_conditional_losses_35920�
���
FullArgSpec
args�
jself
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
annotations� *
 
�2�
(__inference_conv3d_2_layer_call_fn_35929�
���
FullArgSpec
args�
jself
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
annotations� *
 
�2�
F__inference_normalize_2_layer_call_and_return_conditional_losses_35985
F__inference_normalize_2_layer_call_and_return_conditional_losses_35965
F__inference_normalize_2_layer_call_and_return_conditional_losses_36067
F__inference_normalize_2_layer_call_and_return_conditional_losses_36047�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
+__inference_normalize_2_layer_call_fn_36011
+__inference_normalize_2_layer_call_fn_35998
+__inference_normalize_2_layer_call_fn_36093
+__inference_normalize_2_layer_call_fn_36080�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
C__inference_conv3d_3_layer_call_and_return_conditional_losses_36104�
���
FullArgSpec
args�
jself
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
annotations� *
 
�2�
(__inference_conv3d_3_layer_call_fn_36113�
���
FullArgSpec
args�
jself
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
annotations� *
 
�2�
B__inference_flatten_layer_call_and_return_conditional_losses_36119�
���
FullArgSpec
args�
jself
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
annotations� *
 
�2�
'__inference_flatten_layer_call_fn_36124�
���
FullArgSpec
args�
jself
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
annotations� *
 
�2�
@__inference_theta_layer_call_and_return_conditional_losses_36135�
���
FullArgSpec
args�
jself
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
annotations� *
 
�2�
%__inference_theta_layer_call_fn_36144�
���
FullArgSpec
args�
jself
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
annotations� *
 
�B�
#__inference_signature_wrapper_35487input_7"�
���
FullArgSpec
args� 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 �
 __inference__wrapped_model_34649�(%'&-.78=�:
3�0
.�+
input_7���������==�
� ".�+
)
theta �
theta�����������
C__inference_conv3d_1_layer_call_and_return_conditional_losses_35736u<�9
2�/
-�*
inputs���������==�
� "1�.
'�$
0���������K
� �
(__inference_conv3d_1_layer_call_fn_35745h<�9
2�/
-�*
inputs���������==�
� "$�!���������K�
C__inference_conv3d_2_layer_call_and_return_conditional_losses_35920t;�8
1�.
,�)
inputs���������K
� "1�.
'�$
0���������% 
� �
(__inference_conv3d_2_layer_call_fn_35929g;�8
1�.
,�)
inputs���������K
� "$�!���������% �
C__inference_conv3d_3_layer_call_and_return_conditional_losses_36104t-.;�8
1�.
,�)
inputs���������% 
� "1�.
'�$
0���������@
� �
(__inference_conv3d_3_layer_call_fn_36113g-.;�8
1�.
,�)
inputs���������% 
� "$�!���������@�
B__inference_flatten_layer_call_and_return_conditional_losses_36119f;�8
1�.
,�)
inputs���������@
� "'�$
�
0�����������
� �
'__inference_flatten_layer_call_fn_36124Y;�8
1�.
,�)
inputs���������@
� "�������������
B__inference_model_6_layer_call_and_return_conditional_losses_35238�'(%&-.78E�B
;�8
.�+
input_7���������==�
p

 
� "&�#
�
0����������
� �
B__inference_model_6_layer_call_and_return_conditional_losses_35281�(%'&-.78E�B
;�8
.�+
input_7���������==�
p 

 
� "&�#
�
0����������
� �
B__inference_model_6_layer_call_and_return_conditional_losses_35585�'(%&-.78D�A
:�7
-�*
inputs���������==�
p

 
� "&�#
�
0����������
� �
B__inference_model_6_layer_call_and_return_conditional_losses_35651�(%'&-.78D�A
:�7
-�*
inputs���������==�
p 

 
� "&�#
�
0����������
� �
'__inference_model_6_layer_call_fn_35362t'(%&-.78E�B
;�8
.�+
input_7���������==�
p

 
� "������������
'__inference_model_6_layer_call_fn_35442t(%'&-.78E�B
;�8
.�+
input_7���������==�
p 

 
� "������������
'__inference_model_6_layer_call_fn_35688s'(%&-.78D�A
:�7
-�*
inputs���������==�
p

 
� "������������
'__inference_model_6_layer_call_fn_35725s(%'&-.78D�A
:�7
-�*
inputs���������==�
p 

 
� "������������
F__inference_normalize_1_layer_call_and_return_conditional_losses_35781�Z�W
P�M
G�D
inputs8������������������������������������
p
� "L�I
B�?
08������������������������������������
� �
F__inference_normalize_1_layer_call_and_return_conditional_losses_35801�Z�W
P�M
G�D
inputs8������������������������������������
p 
� "L�I
B�?
08������������������������������������
� �
F__inference_normalize_1_layer_call_and_return_conditional_losses_35863z?�<
5�2
,�)
inputs���������K
p
� "1�.
'�$
0���������K
� �
F__inference_normalize_1_layer_call_and_return_conditional_losses_35883z?�<
5�2
,�)
inputs���������K
p 
� "1�.
'�$
0���������K
� �
+__inference_normalize_1_layer_call_fn_35814�Z�W
P�M
G�D
inputs8������������������������������������
p
� "?�<8�������������������������������������
+__inference_normalize_1_layer_call_fn_35827�Z�W
P�M
G�D
inputs8������������������������������������
p 
� "?�<8�������������������������������������
+__inference_normalize_1_layer_call_fn_35896m?�<
5�2
,�)
inputs���������K
p
� "$�!���������K�
+__inference_normalize_1_layer_call_fn_35909m?�<
5�2
,�)
inputs���������K
p 
� "$�!���������K�
F__inference_normalize_2_layer_call_and_return_conditional_losses_35965�'(%&Z�W
P�M
G�D
inputs8������������������������������������ 
p
� "L�I
B�?
08������������������������������������ 
� �
F__inference_normalize_2_layer_call_and_return_conditional_losses_35985�(%'&Z�W
P�M
G�D
inputs8������������������������������������ 
p 
� "L�I
B�?
08������������������������������������ 
� �
F__inference_normalize_2_layer_call_and_return_conditional_losses_36047z'(%&?�<
5�2
,�)
inputs���������% 
p
� "1�.
'�$
0���������% 
� �
F__inference_normalize_2_layer_call_and_return_conditional_losses_36067z(%'&?�<
5�2
,�)
inputs���������% 
p 
� "1�.
'�$
0���������% 
� �
+__inference_normalize_2_layer_call_fn_35998�'(%&Z�W
P�M
G�D
inputs8������������������������������������ 
p
� "?�<8������������������������������������ �
+__inference_normalize_2_layer_call_fn_36011�(%'&Z�W
P�M
G�D
inputs8������������������������������������ 
p 
� "?�<8������������������������������������ �
+__inference_normalize_2_layer_call_fn_36080m'(%&?�<
5�2
,�)
inputs���������% 
p
� "$�!���������% �
+__inference_normalize_2_layer_call_fn_36093m(%'&?�<
5�2
,�)
inputs���������% 
p 
� "$�!���������% �
#__inference_signature_wrapper_35487�(%'&-.78H�E
� 
>�;
9
input_7.�+
input_7���������==�".�+
)
theta �
theta�����������
@__inference_theta_layer_call_and_return_conditional_losses_36135_781�.
'�$
"�
inputs�����������
� "&�#
�
0����������
� {
%__inference_theta_layer_call_fn_36144R781�.
'�$
"�
inputs�����������
� "�����������