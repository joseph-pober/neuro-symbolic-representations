��
��
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( �
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
�
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( �
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
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
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
0
Sigmoid
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
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
�
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �"serve*2.10.02v2.10.0-rc3-6-g359c3cdfc5f8��
p
p3_code/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namep3_code/bias
i
 p3_code/bias/Read/ReadVariableOpReadVariableOpp3_code/bias*
_output_shapes
:*
dtype0
x
p3_code/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*
shared_namep3_code/kernel
q
"p3_code/kernel/Read/ReadVariableOpReadVariableOpp3_code/kernel*
_output_shapes

:*
dtype0
p
p2_code/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namep2_code/bias
i
 p2_code/bias/Read/ReadVariableOpReadVariableOpp2_code/bias*
_output_shapes
:*
dtype0
x
p2_code/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*
shared_namep2_code/kernel
q
"p2_code/kernel/Read/ReadVariableOpReadVariableOpp2_code/kernel*
_output_shapes

:*
dtype0
p
p1_code/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namep1_code/bias
i
 p1_code/bias/Read/ReadVariableOpReadVariableOpp1_code/bias*
_output_shapes
:*
dtype0
x
p1_code/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*
shared_namep1_code/kernel
q
"p1_code/kernel/Read/ReadVariableOpReadVariableOpp1_code/kernel*
_output_shapes

:*
dtype0
j
	p3_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name	p3_1/bias
c
p3_1/bias/Read/ReadVariableOpReadVariableOp	p3_1/bias*
_output_shapes
:*
dtype0
r
p3_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *
shared_namep3_1/kernel
k
p3_1/kernel/Read/ReadVariableOpReadVariableOpp3_1/kernel*
_output_shapes

: *
dtype0
j
	p2_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name	p2_1/bias
c
p2_1/bias/Read/ReadVariableOpReadVariableOp	p2_1/bias*
_output_shapes
:*
dtype0
r
p2_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *
shared_namep2_1/kernel
k
p2_1/kernel/Read/ReadVariableOpReadVariableOpp2_1/kernel*
_output_shapes

: *
dtype0
j
	p1_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name	p1_1/bias
c
p1_1/bias/Read/ReadVariableOpReadVariableOp	p1_1/bias*
_output_shapes
:*
dtype0
r
p1_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *
shared_namep1_1/kernel
k
p1_1/kernel/Read/ReadVariableOpReadVariableOpp1_1/kernel*
_output_shapes

: *
dtype0
�
serving_default_original_codePlaceholder*'
_output_shapes
:��������� *
dtype0*
shape:��������� 
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_original_codep3_1/kernel	p3_1/biasp2_1/kernel	p2_1/biasp1_1/kernel	p1_1/biasp3_code/kernelp3_code/biasp2_code/kernelp2_code/biasp1_code/kernelp1_code/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:���������:���������:���������*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *,
f'R%
#__inference_signature_wrapper_22613

NoOpNoOp
�,
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�,
value�+B�+ B�+
�
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
layer_with_weights-5
layer-6
	variables
	trainable_variables

regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures*
* 
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
bias*
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
bias*
�
 	variables
!trainable_variables
"regularization_losses
#	keras_api
$__call__
*%&call_and_return_all_conditional_losses

&kernel
'bias*
�
(	variables
)trainable_variables
*regularization_losses
+	keras_api
,__call__
*-&call_and_return_all_conditional_losses

.kernel
/bias*
�
0	variables
1trainable_variables
2regularization_losses
3	keras_api
4__call__
*5&call_and_return_all_conditional_losses

6kernel
7bias*
�
8	variables
9trainable_variables
:regularization_losses
;	keras_api
<__call__
*=&call_and_return_all_conditional_losses

>kernel
?bias*
Z
0
1
2
3
&4
'5
.6
/7
68
79
>10
?11*
Z
0
1
2
3
&4
'5
.6
/7
68
79
>10
?11*
* 
�
@non_trainable_variables

Alayers
Bmetrics
Clayer_regularization_losses
Dlayer_metrics
	variables
	trainable_variables

regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
6
Etrace_0
Ftrace_1
Gtrace_2
Htrace_3* 
6
Itrace_0
Jtrace_1
Ktrace_2
Ltrace_3* 
* 

Mserving_default* 

0
1*

0
1*
* 
�
Nnon_trainable_variables

Olayers
Pmetrics
Qlayer_regularization_losses
Rlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

Strace_0* 

Ttrace_0* 
[U
VARIABLE_VALUEp1_1/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*
WQ
VARIABLE_VALUE	p1_1/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE*

0
1*

0
1*
* 
�
Unon_trainable_variables

Vlayers
Wmetrics
Xlayer_regularization_losses
Ylayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

Ztrace_0* 

[trace_0* 
[U
VARIABLE_VALUEp2_1/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
WQ
VARIABLE_VALUE	p2_1/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*

&0
'1*

&0
'1*
* 
�
\non_trainable_variables

]layers
^metrics
_layer_regularization_losses
`layer_metrics
 	variables
!trainable_variables
"regularization_losses
$__call__
*%&call_and_return_all_conditional_losses
&%"call_and_return_conditional_losses*

atrace_0* 

btrace_0* 
[U
VARIABLE_VALUEp3_1/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
WQ
VARIABLE_VALUE	p3_1/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*

.0
/1*

.0
/1*
* 
�
cnon_trainable_variables

dlayers
emetrics
flayer_regularization_losses
glayer_metrics
(	variables
)trainable_variables
*regularization_losses
,__call__
*-&call_and_return_all_conditional_losses
&-"call_and_return_conditional_losses*

htrace_0* 

itrace_0* 
^X
VARIABLE_VALUEp1_code/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEp1_code/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE*

60
71*

60
71*
* 
�
jnon_trainable_variables

klayers
lmetrics
mlayer_regularization_losses
nlayer_metrics
0	variables
1trainable_variables
2regularization_losses
4__call__
*5&call_and_return_all_conditional_losses
&5"call_and_return_conditional_losses*

otrace_0* 

ptrace_0* 
^X
VARIABLE_VALUEp2_code/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEp2_code/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE*

>0
?1*

>0
?1*
* 
�
qnon_trainable_variables

rlayers
smetrics
tlayer_regularization_losses
ulayer_metrics
8	variables
9trainable_variables
:regularization_losses
<__call__
*=&call_and_return_all_conditional_losses
&="call_and_return_conditional_losses*

vtrace_0* 

wtrace_0* 
^X
VARIABLE_VALUEp3_code/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEp3_code/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
5
0
1
2
3
4
5
6*
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
* 
* 
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenamep1_1/kernel/Read/ReadVariableOpp1_1/bias/Read/ReadVariableOpp2_1/kernel/Read/ReadVariableOpp2_1/bias/Read/ReadVariableOpp3_1/kernel/Read/ReadVariableOpp3_1/bias/Read/ReadVariableOp"p1_code/kernel/Read/ReadVariableOp p1_code/bias/Read/ReadVariableOp"p2_code/kernel/Read/ReadVariableOp p2_code/bias/Read/ReadVariableOp"p3_code/kernel/Read/ReadVariableOp p3_code/bias/Read/ReadVariableOpConst*
Tin
2*
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
__inference__traced_save_22956
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamep1_1/kernel	p1_1/biasp2_1/kernel	p2_1/biasp3_1/kernel	p3_1/biasp1_code/kernelp1_code/biasp2_code/kernelp2_code/biasp3_code/kernelp3_code/bias*
Tin
2*
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
!__inference__traced_restore_23002��
�
�
'__inference_Encoder_layer_call_fn_22679

inputs
unknown: 
	unknown_0:
	unknown_1: 
	unknown_2:
	unknown_3: 
	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10:
identity

identity_1

identity_2��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:���������:���������:���������*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_Encoder_layer_call_and_return_conditional_losses_22442o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:���������q

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:��������� : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�

�
?__inference_p2_1_layer_call_and_return_conditional_losses_22203

inputs0
matmul_readvariableop_resource: -
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������V
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:���������Z
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:��������� : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�!
�
B__inference_Encoder_layer_call_and_return_conditional_losses_22542
original_code

p3_1_22509: 

p3_1_22511:

p2_1_22514: 

p2_1_22516:

p1_1_22519: 

p1_1_22521:
p3_code_22524:
p3_code_22526:
p2_code_22529:
p2_code_22531:
p1_code_22534:
p1_code_22536:
identity

identity_1

identity_2��p1_1/StatefulPartitionedCall�p1_code/StatefulPartitionedCall�p2_1/StatefulPartitionedCall�p2_code/StatefulPartitionedCall�p3_1/StatefulPartitionedCall�p3_code/StatefulPartitionedCall�
p3_1/StatefulPartitionedCallStatefulPartitionedCalloriginal_code
p3_1_22509
p3_1_22511*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *H
fCRA
?__inference_p3_1_layer_call_and_return_conditional_losses_22186�
p2_1/StatefulPartitionedCallStatefulPartitionedCalloriginal_code
p2_1_22514
p2_1_22516*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *H
fCRA
?__inference_p2_1_layer_call_and_return_conditional_losses_22203�
p1_1/StatefulPartitionedCallStatefulPartitionedCalloriginal_code
p1_1_22519
p1_1_22521*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *H
fCRA
?__inference_p1_1_layer_call_and_return_conditional_losses_22220�
p3_code/StatefulPartitionedCallStatefulPartitionedCall%p3_1/StatefulPartitionedCall:output:0p3_code_22524p3_code_22526*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_p3_code_layer_call_and_return_conditional_losses_22237�
p2_code/StatefulPartitionedCallStatefulPartitionedCall%p2_1/StatefulPartitionedCall:output:0p2_code_22529p2_code_22531*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_p2_code_layer_call_and_return_conditional_losses_22254�
p1_code/StatefulPartitionedCallStatefulPartitionedCall%p1_1/StatefulPartitionedCall:output:0p1_code_22534p1_code_22536*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_p1_code_layer_call_and_return_conditional_losses_22271w
IdentityIdentity(p1_code/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������y

Identity_1Identity(p2_code/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������y

Identity_2Identity(p3_code/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^p1_1/StatefulPartitionedCall ^p1_code/StatefulPartitionedCall^p2_1/StatefulPartitionedCall ^p2_code/StatefulPartitionedCall^p3_1/StatefulPartitionedCall ^p3_code/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:��������� : : : : : : : : : : : : 2<
p1_1/StatefulPartitionedCallp1_1/StatefulPartitionedCall2B
p1_code/StatefulPartitionedCallp1_code/StatefulPartitionedCall2<
p2_1/StatefulPartitionedCallp2_1/StatefulPartitionedCall2B
p2_code/StatefulPartitionedCallp2_code/StatefulPartitionedCall2<
p3_1/StatefulPartitionedCallp3_1/StatefulPartitionedCall2B
p3_code/StatefulPartitionedCallp3_code/StatefulPartitionedCall:V R
'
_output_shapes
:��������� 
'
_user_specified_nameoriginal_code
�

�
B__inference_p2_code_layer_call_and_return_conditional_losses_22875

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������V
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:���������Z
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
$__inference_p2_1_layer_call_fn_22804

inputs
unknown: 
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *H
fCRA
?__inference_p2_1_layer_call_and_return_conditional_losses_22203o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:��������� : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�

�
B__inference_p3_code_layer_call_and_return_conditional_losses_22895

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������V
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:���������Z
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�<
�

 __inference__wrapped_model_22168
original_code=
+encoder_p3_1_matmul_readvariableop_resource: :
,encoder_p3_1_biasadd_readvariableop_resource:=
+encoder_p2_1_matmul_readvariableop_resource: :
,encoder_p2_1_biasadd_readvariableop_resource:=
+encoder_p1_1_matmul_readvariableop_resource: :
,encoder_p1_1_biasadd_readvariableop_resource:@
.encoder_p3_code_matmul_readvariableop_resource:=
/encoder_p3_code_biasadd_readvariableop_resource:@
.encoder_p2_code_matmul_readvariableop_resource:=
/encoder_p2_code_biasadd_readvariableop_resource:@
.encoder_p1_code_matmul_readvariableop_resource:=
/encoder_p1_code_biasadd_readvariableop_resource:
identity

identity_1

identity_2��#Encoder/p1_1/BiasAdd/ReadVariableOp�"Encoder/p1_1/MatMul/ReadVariableOp�&Encoder/p1_code/BiasAdd/ReadVariableOp�%Encoder/p1_code/MatMul/ReadVariableOp�#Encoder/p2_1/BiasAdd/ReadVariableOp�"Encoder/p2_1/MatMul/ReadVariableOp�&Encoder/p2_code/BiasAdd/ReadVariableOp�%Encoder/p2_code/MatMul/ReadVariableOp�#Encoder/p3_1/BiasAdd/ReadVariableOp�"Encoder/p3_1/MatMul/ReadVariableOp�&Encoder/p3_code/BiasAdd/ReadVariableOp�%Encoder/p3_code/MatMul/ReadVariableOp�
"Encoder/p3_1/MatMul/ReadVariableOpReadVariableOp+encoder_p3_1_matmul_readvariableop_resource*
_output_shapes

: *
dtype0�
Encoder/p3_1/MatMulMatMuloriginal_code*Encoder/p3_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
#Encoder/p3_1/BiasAdd/ReadVariableOpReadVariableOp,encoder_p3_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
Encoder/p3_1/BiasAddBiasAddEncoder/p3_1/MatMul:product:0+Encoder/p3_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������p
Encoder/p3_1/SigmoidSigmoidEncoder/p3_1/BiasAdd:output:0*
T0*'
_output_shapes
:����������
"Encoder/p2_1/MatMul/ReadVariableOpReadVariableOp+encoder_p2_1_matmul_readvariableop_resource*
_output_shapes

: *
dtype0�
Encoder/p2_1/MatMulMatMuloriginal_code*Encoder/p2_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
#Encoder/p2_1/BiasAdd/ReadVariableOpReadVariableOp,encoder_p2_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
Encoder/p2_1/BiasAddBiasAddEncoder/p2_1/MatMul:product:0+Encoder/p2_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������p
Encoder/p2_1/SigmoidSigmoidEncoder/p2_1/BiasAdd:output:0*
T0*'
_output_shapes
:����������
"Encoder/p1_1/MatMul/ReadVariableOpReadVariableOp+encoder_p1_1_matmul_readvariableop_resource*
_output_shapes

: *
dtype0�
Encoder/p1_1/MatMulMatMuloriginal_code*Encoder/p1_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
#Encoder/p1_1/BiasAdd/ReadVariableOpReadVariableOp,encoder_p1_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
Encoder/p1_1/BiasAddBiasAddEncoder/p1_1/MatMul:product:0+Encoder/p1_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������p
Encoder/p1_1/SigmoidSigmoidEncoder/p1_1/BiasAdd:output:0*
T0*'
_output_shapes
:����������
%Encoder/p3_code/MatMul/ReadVariableOpReadVariableOp.encoder_p3_code_matmul_readvariableop_resource*
_output_shapes

:*
dtype0�
Encoder/p3_code/MatMulMatMulEncoder/p3_1/Sigmoid:y:0-Encoder/p3_code/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
&Encoder/p3_code/BiasAdd/ReadVariableOpReadVariableOp/encoder_p3_code_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
Encoder/p3_code/BiasAddBiasAdd Encoder/p3_code/MatMul:product:0.Encoder/p3_code/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������v
Encoder/p3_code/SigmoidSigmoid Encoder/p3_code/BiasAdd:output:0*
T0*'
_output_shapes
:����������
%Encoder/p2_code/MatMul/ReadVariableOpReadVariableOp.encoder_p2_code_matmul_readvariableop_resource*
_output_shapes

:*
dtype0�
Encoder/p2_code/MatMulMatMulEncoder/p2_1/Sigmoid:y:0-Encoder/p2_code/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
&Encoder/p2_code/BiasAdd/ReadVariableOpReadVariableOp/encoder_p2_code_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
Encoder/p2_code/BiasAddBiasAdd Encoder/p2_code/MatMul:product:0.Encoder/p2_code/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������v
Encoder/p2_code/SigmoidSigmoid Encoder/p2_code/BiasAdd:output:0*
T0*'
_output_shapes
:����������
%Encoder/p1_code/MatMul/ReadVariableOpReadVariableOp.encoder_p1_code_matmul_readvariableop_resource*
_output_shapes

:*
dtype0�
Encoder/p1_code/MatMulMatMulEncoder/p1_1/Sigmoid:y:0-Encoder/p1_code/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
&Encoder/p1_code/BiasAdd/ReadVariableOpReadVariableOp/encoder_p1_code_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
Encoder/p1_code/BiasAddBiasAdd Encoder/p1_code/MatMul:product:0.Encoder/p1_code/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������v
Encoder/p1_code/SigmoidSigmoid Encoder/p1_code/BiasAdd:output:0*
T0*'
_output_shapes
:���������j
IdentityIdentityEncoder/p1_code/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:���������l

Identity_1IdentityEncoder/p2_code/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:���������l

Identity_2IdentityEncoder/p3_code/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp$^Encoder/p1_1/BiasAdd/ReadVariableOp#^Encoder/p1_1/MatMul/ReadVariableOp'^Encoder/p1_code/BiasAdd/ReadVariableOp&^Encoder/p1_code/MatMul/ReadVariableOp$^Encoder/p2_1/BiasAdd/ReadVariableOp#^Encoder/p2_1/MatMul/ReadVariableOp'^Encoder/p2_code/BiasAdd/ReadVariableOp&^Encoder/p2_code/MatMul/ReadVariableOp$^Encoder/p3_1/BiasAdd/ReadVariableOp#^Encoder/p3_1/MatMul/ReadVariableOp'^Encoder/p3_code/BiasAdd/ReadVariableOp&^Encoder/p3_code/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:��������� : : : : : : : : : : : : 2J
#Encoder/p1_1/BiasAdd/ReadVariableOp#Encoder/p1_1/BiasAdd/ReadVariableOp2H
"Encoder/p1_1/MatMul/ReadVariableOp"Encoder/p1_1/MatMul/ReadVariableOp2P
&Encoder/p1_code/BiasAdd/ReadVariableOp&Encoder/p1_code/BiasAdd/ReadVariableOp2N
%Encoder/p1_code/MatMul/ReadVariableOp%Encoder/p1_code/MatMul/ReadVariableOp2J
#Encoder/p2_1/BiasAdd/ReadVariableOp#Encoder/p2_1/BiasAdd/ReadVariableOp2H
"Encoder/p2_1/MatMul/ReadVariableOp"Encoder/p2_1/MatMul/ReadVariableOp2P
&Encoder/p2_code/BiasAdd/ReadVariableOp&Encoder/p2_code/BiasAdd/ReadVariableOp2N
%Encoder/p2_code/MatMul/ReadVariableOp%Encoder/p2_code/MatMul/ReadVariableOp2J
#Encoder/p3_1/BiasAdd/ReadVariableOp#Encoder/p3_1/BiasAdd/ReadVariableOp2H
"Encoder/p3_1/MatMul/ReadVariableOp"Encoder/p3_1/MatMul/ReadVariableOp2P
&Encoder/p3_code/BiasAdd/ReadVariableOp&Encoder/p3_code/BiasAdd/ReadVariableOp2N
%Encoder/p3_code/MatMul/ReadVariableOp%Encoder/p3_code/MatMul/ReadVariableOp:V R
'
_output_shapes
:��������� 
'
_user_specified_nameoriginal_code
�

�
?__inference_p2_1_layer_call_and_return_conditional_losses_22815

inputs0
matmul_readvariableop_resource: -
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������V
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:���������Z
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:��������� : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
�
$__inference_p3_1_layer_call_fn_22824

inputs
unknown: 
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *H
fCRA
?__inference_p3_1_layer_call_and_return_conditional_losses_22186o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:��������� : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�!
�
B__inference_Encoder_layer_call_and_return_conditional_losses_22442

inputs

p3_1_22409: 

p3_1_22411:

p2_1_22414: 

p2_1_22416:

p1_1_22419: 

p1_1_22421:
p3_code_22424:
p3_code_22426:
p2_code_22429:
p2_code_22431:
p1_code_22434:
p1_code_22436:
identity

identity_1

identity_2��p1_1/StatefulPartitionedCall�p1_code/StatefulPartitionedCall�p2_1/StatefulPartitionedCall�p2_code/StatefulPartitionedCall�p3_1/StatefulPartitionedCall�p3_code/StatefulPartitionedCall�
p3_1/StatefulPartitionedCallStatefulPartitionedCallinputs
p3_1_22409
p3_1_22411*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *H
fCRA
?__inference_p3_1_layer_call_and_return_conditional_losses_22186�
p2_1/StatefulPartitionedCallStatefulPartitionedCallinputs
p2_1_22414
p2_1_22416*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *H
fCRA
?__inference_p2_1_layer_call_and_return_conditional_losses_22203�
p1_1/StatefulPartitionedCallStatefulPartitionedCallinputs
p1_1_22419
p1_1_22421*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *H
fCRA
?__inference_p1_1_layer_call_and_return_conditional_losses_22220�
p3_code/StatefulPartitionedCallStatefulPartitionedCall%p3_1/StatefulPartitionedCall:output:0p3_code_22424p3_code_22426*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_p3_code_layer_call_and_return_conditional_losses_22237�
p2_code/StatefulPartitionedCallStatefulPartitionedCall%p2_1/StatefulPartitionedCall:output:0p2_code_22429p2_code_22431*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_p2_code_layer_call_and_return_conditional_losses_22254�
p1_code/StatefulPartitionedCallStatefulPartitionedCall%p1_1/StatefulPartitionedCall:output:0p1_code_22434p1_code_22436*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_p1_code_layer_call_and_return_conditional_losses_22271w
IdentityIdentity(p1_code/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������y

Identity_1Identity(p2_code/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������y

Identity_2Identity(p3_code/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^p1_1/StatefulPartitionedCall ^p1_code/StatefulPartitionedCall^p2_1/StatefulPartitionedCall ^p2_code/StatefulPartitionedCall^p3_1/StatefulPartitionedCall ^p3_code/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:��������� : : : : : : : : : : : : 2<
p1_1/StatefulPartitionedCallp1_1/StatefulPartitionedCall2B
p1_code/StatefulPartitionedCallp1_code/StatefulPartitionedCall2<
p2_1/StatefulPartitionedCallp2_1/StatefulPartitionedCall2B
p2_code/StatefulPartitionedCallp2_code/StatefulPartitionedCall2<
p3_1/StatefulPartitionedCallp3_1/StatefulPartitionedCall2B
p3_code/StatefulPartitionedCallp3_code/StatefulPartitionedCall:O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
�
'__inference_Encoder_layer_call_fn_22646

inputs
unknown: 
	unknown_0:
	unknown_1: 
	unknown_2:
	unknown_3: 
	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10:
identity

identity_1

identity_2��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:���������:���������:���������*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_Encoder_layer_call_and_return_conditional_losses_22280o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:���������q

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:��������� : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�

�
B__inference_p1_code_layer_call_and_return_conditional_losses_22855

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������V
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:���������Z
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
'__inference_Encoder_layer_call_fn_22506
original_code
unknown: 
	unknown_0:
	unknown_1: 
	unknown_2:
	unknown_3: 
	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10:
identity

identity_1

identity_2��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCalloriginal_codeunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:���������:���������:���������*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_Encoder_layer_call_and_return_conditional_losses_22442o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:���������q

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:��������� : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:V R
'
_output_shapes
:��������� 
'
_user_specified_nameoriginal_code
�"
�
__inference__traced_save_22956
file_prefix*
&savev2_p1_1_kernel_read_readvariableop(
$savev2_p1_1_bias_read_readvariableop*
&savev2_p2_1_kernel_read_readvariableop(
$savev2_p2_1_bias_read_readvariableop*
&savev2_p3_1_kernel_read_readvariableop(
$savev2_p3_1_bias_read_readvariableop-
)savev2_p1_code_kernel_read_readvariableop+
'savev2_p1_code_bias_read_readvariableop-
)savev2_p2_code_kernel_read_readvariableop+
'savev2_p2_code_bias_read_readvariableop-
)savev2_p3_code_kernel_read_readvariableop+
'savev2_p3_code_bias_read_readvariableop
savev2_const

identity_1��MergeV2Checkpointsw
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
value	B : �
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: �
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*-
value$B"B B B B B B B B B B B B B �
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0&savev2_p1_1_kernel_read_readvariableop$savev2_p1_1_bias_read_readvariableop&savev2_p2_1_kernel_read_readvariableop$savev2_p2_1_bias_read_readvariableop&savev2_p3_1_kernel_read_readvariableop$savev2_p3_1_bias_read_readvariableop)savev2_p1_code_kernel_read_readvariableop'savev2_p1_code_bias_read_readvariableop)savev2_p2_code_kernel_read_readvariableop'savev2_p2_code_bias_read_readvariableop)savev2_p3_code_kernel_read_readvariableop'savev2_p3_code_bias_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *
dtypes
2�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*w
_input_shapesf
d: : :: :: :::::::: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

: : 

_output_shapes
::$ 

_output_shapes

: : 

_output_shapes
::$ 

_output_shapes

: : 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::$	 

_output_shapes

:: 


_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::

_output_shapes
: 
�4
�	
B__inference_Encoder_layer_call_and_return_conditional_losses_22775

inputs5
#p3_1_matmul_readvariableop_resource: 2
$p3_1_biasadd_readvariableop_resource:5
#p2_1_matmul_readvariableop_resource: 2
$p2_1_biasadd_readvariableop_resource:5
#p1_1_matmul_readvariableop_resource: 2
$p1_1_biasadd_readvariableop_resource:8
&p3_code_matmul_readvariableop_resource:5
'p3_code_biasadd_readvariableop_resource:8
&p2_code_matmul_readvariableop_resource:5
'p2_code_biasadd_readvariableop_resource:8
&p1_code_matmul_readvariableop_resource:5
'p1_code_biasadd_readvariableop_resource:
identity

identity_1

identity_2��p1_1/BiasAdd/ReadVariableOp�p1_1/MatMul/ReadVariableOp�p1_code/BiasAdd/ReadVariableOp�p1_code/MatMul/ReadVariableOp�p2_1/BiasAdd/ReadVariableOp�p2_1/MatMul/ReadVariableOp�p2_code/BiasAdd/ReadVariableOp�p2_code/MatMul/ReadVariableOp�p3_1/BiasAdd/ReadVariableOp�p3_1/MatMul/ReadVariableOp�p3_code/BiasAdd/ReadVariableOp�p3_code/MatMul/ReadVariableOp~
p3_1/MatMul/ReadVariableOpReadVariableOp#p3_1_matmul_readvariableop_resource*
_output_shapes

: *
dtype0s
p3_1/MatMulMatMulinputs"p3_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������|
p3_1/BiasAdd/ReadVariableOpReadVariableOp$p3_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
p3_1/BiasAddBiasAddp3_1/MatMul:product:0#p3_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������`
p3_1/SigmoidSigmoidp3_1/BiasAdd:output:0*
T0*'
_output_shapes
:���������~
p2_1/MatMul/ReadVariableOpReadVariableOp#p2_1_matmul_readvariableop_resource*
_output_shapes

: *
dtype0s
p2_1/MatMulMatMulinputs"p2_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������|
p2_1/BiasAdd/ReadVariableOpReadVariableOp$p2_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
p2_1/BiasAddBiasAddp2_1/MatMul:product:0#p2_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������`
p2_1/SigmoidSigmoidp2_1/BiasAdd:output:0*
T0*'
_output_shapes
:���������~
p1_1/MatMul/ReadVariableOpReadVariableOp#p1_1_matmul_readvariableop_resource*
_output_shapes

: *
dtype0s
p1_1/MatMulMatMulinputs"p1_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������|
p1_1/BiasAdd/ReadVariableOpReadVariableOp$p1_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
p1_1/BiasAddBiasAddp1_1/MatMul:product:0#p1_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������`
p1_1/SigmoidSigmoidp1_1/BiasAdd:output:0*
T0*'
_output_shapes
:����������
p3_code/MatMul/ReadVariableOpReadVariableOp&p3_code_matmul_readvariableop_resource*
_output_shapes

:*
dtype0�
p3_code/MatMulMatMulp3_1/Sigmoid:y:0%p3_code/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
p3_code/BiasAdd/ReadVariableOpReadVariableOp'p3_code_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
p3_code/BiasAddBiasAddp3_code/MatMul:product:0&p3_code/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������f
p3_code/SigmoidSigmoidp3_code/BiasAdd:output:0*
T0*'
_output_shapes
:����������
p2_code/MatMul/ReadVariableOpReadVariableOp&p2_code_matmul_readvariableop_resource*
_output_shapes

:*
dtype0�
p2_code/MatMulMatMulp2_1/Sigmoid:y:0%p2_code/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
p2_code/BiasAdd/ReadVariableOpReadVariableOp'p2_code_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
p2_code/BiasAddBiasAddp2_code/MatMul:product:0&p2_code/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������f
p2_code/SigmoidSigmoidp2_code/BiasAdd:output:0*
T0*'
_output_shapes
:����������
p1_code/MatMul/ReadVariableOpReadVariableOp&p1_code_matmul_readvariableop_resource*
_output_shapes

:*
dtype0�
p1_code/MatMulMatMulp1_1/Sigmoid:y:0%p1_code/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
p1_code/BiasAdd/ReadVariableOpReadVariableOp'p1_code_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
p1_code/BiasAddBiasAddp1_code/MatMul:product:0&p1_code/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������f
p1_code/SigmoidSigmoidp1_code/BiasAdd:output:0*
T0*'
_output_shapes
:���������b
IdentityIdentityp1_code/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:���������d

Identity_1Identityp2_code/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:���������d

Identity_2Identityp3_code/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^p1_1/BiasAdd/ReadVariableOp^p1_1/MatMul/ReadVariableOp^p1_code/BiasAdd/ReadVariableOp^p1_code/MatMul/ReadVariableOp^p2_1/BiasAdd/ReadVariableOp^p2_1/MatMul/ReadVariableOp^p2_code/BiasAdd/ReadVariableOp^p2_code/MatMul/ReadVariableOp^p3_1/BiasAdd/ReadVariableOp^p3_1/MatMul/ReadVariableOp^p3_code/BiasAdd/ReadVariableOp^p3_code/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:��������� : : : : : : : : : : : : 2:
p1_1/BiasAdd/ReadVariableOpp1_1/BiasAdd/ReadVariableOp28
p1_1/MatMul/ReadVariableOpp1_1/MatMul/ReadVariableOp2@
p1_code/BiasAdd/ReadVariableOpp1_code/BiasAdd/ReadVariableOp2>
p1_code/MatMul/ReadVariableOpp1_code/MatMul/ReadVariableOp2:
p2_1/BiasAdd/ReadVariableOpp2_1/BiasAdd/ReadVariableOp28
p2_1/MatMul/ReadVariableOpp2_1/MatMul/ReadVariableOp2@
p2_code/BiasAdd/ReadVariableOpp2_code/BiasAdd/ReadVariableOp2>
p2_code/MatMul/ReadVariableOpp2_code/MatMul/ReadVariableOp2:
p3_1/BiasAdd/ReadVariableOpp3_1/BiasAdd/ReadVariableOp28
p3_1/MatMul/ReadVariableOpp3_1/MatMul/ReadVariableOp2@
p3_code/BiasAdd/ReadVariableOpp3_code/BiasAdd/ReadVariableOp2>
p3_code/MatMul/ReadVariableOpp3_code/MatMul/ReadVariableOp:O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�

�
?__inference_p1_1_layer_call_and_return_conditional_losses_22795

inputs0
matmul_readvariableop_resource: -
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������V
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:���������Z
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:��������� : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�

�
?__inference_p1_1_layer_call_and_return_conditional_losses_22220

inputs0
matmul_readvariableop_resource: -
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������V
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:���������Z
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:��������� : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
�
'__inference_p3_code_layer_call_fn_22884

inputs
unknown:
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_p3_code_layer_call_and_return_conditional_losses_22237o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
$__inference_p1_1_layer_call_fn_22784

inputs
unknown: 
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *H
fCRA
?__inference_p1_1_layer_call_and_return_conditional_losses_22220o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:��������� : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�!
�
B__inference_Encoder_layer_call_and_return_conditional_losses_22578
original_code

p3_1_22545: 

p3_1_22547:

p2_1_22550: 

p2_1_22552:

p1_1_22555: 

p1_1_22557:
p3_code_22560:
p3_code_22562:
p2_code_22565:
p2_code_22567:
p1_code_22570:
p1_code_22572:
identity

identity_1

identity_2��p1_1/StatefulPartitionedCall�p1_code/StatefulPartitionedCall�p2_1/StatefulPartitionedCall�p2_code/StatefulPartitionedCall�p3_1/StatefulPartitionedCall�p3_code/StatefulPartitionedCall�
p3_1/StatefulPartitionedCallStatefulPartitionedCalloriginal_code
p3_1_22545
p3_1_22547*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *H
fCRA
?__inference_p3_1_layer_call_and_return_conditional_losses_22186�
p2_1/StatefulPartitionedCallStatefulPartitionedCalloriginal_code
p2_1_22550
p2_1_22552*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *H
fCRA
?__inference_p2_1_layer_call_and_return_conditional_losses_22203�
p1_1/StatefulPartitionedCallStatefulPartitionedCalloriginal_code
p1_1_22555
p1_1_22557*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *H
fCRA
?__inference_p1_1_layer_call_and_return_conditional_losses_22220�
p3_code/StatefulPartitionedCallStatefulPartitionedCall%p3_1/StatefulPartitionedCall:output:0p3_code_22560p3_code_22562*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_p3_code_layer_call_and_return_conditional_losses_22237�
p2_code/StatefulPartitionedCallStatefulPartitionedCall%p2_1/StatefulPartitionedCall:output:0p2_code_22565p2_code_22567*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_p2_code_layer_call_and_return_conditional_losses_22254�
p1_code/StatefulPartitionedCallStatefulPartitionedCall%p1_1/StatefulPartitionedCall:output:0p1_code_22570p1_code_22572*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_p1_code_layer_call_and_return_conditional_losses_22271w
IdentityIdentity(p1_code/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������y

Identity_1Identity(p2_code/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������y

Identity_2Identity(p3_code/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^p1_1/StatefulPartitionedCall ^p1_code/StatefulPartitionedCall^p2_1/StatefulPartitionedCall ^p2_code/StatefulPartitionedCall^p3_1/StatefulPartitionedCall ^p3_code/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:��������� : : : : : : : : : : : : 2<
p1_1/StatefulPartitionedCallp1_1/StatefulPartitionedCall2B
p1_code/StatefulPartitionedCallp1_code/StatefulPartitionedCall2<
p2_1/StatefulPartitionedCallp2_1/StatefulPartitionedCall2B
p2_code/StatefulPartitionedCallp2_code/StatefulPartitionedCall2<
p3_1/StatefulPartitionedCallp3_1/StatefulPartitionedCall2B
p3_code/StatefulPartitionedCallp3_code/StatefulPartitionedCall:V R
'
_output_shapes
:��������� 
'
_user_specified_nameoriginal_code
�4
�	
B__inference_Encoder_layer_call_and_return_conditional_losses_22727

inputs5
#p3_1_matmul_readvariableop_resource: 2
$p3_1_biasadd_readvariableop_resource:5
#p2_1_matmul_readvariableop_resource: 2
$p2_1_biasadd_readvariableop_resource:5
#p1_1_matmul_readvariableop_resource: 2
$p1_1_biasadd_readvariableop_resource:8
&p3_code_matmul_readvariableop_resource:5
'p3_code_biasadd_readvariableop_resource:8
&p2_code_matmul_readvariableop_resource:5
'p2_code_biasadd_readvariableop_resource:8
&p1_code_matmul_readvariableop_resource:5
'p1_code_biasadd_readvariableop_resource:
identity

identity_1

identity_2��p1_1/BiasAdd/ReadVariableOp�p1_1/MatMul/ReadVariableOp�p1_code/BiasAdd/ReadVariableOp�p1_code/MatMul/ReadVariableOp�p2_1/BiasAdd/ReadVariableOp�p2_1/MatMul/ReadVariableOp�p2_code/BiasAdd/ReadVariableOp�p2_code/MatMul/ReadVariableOp�p3_1/BiasAdd/ReadVariableOp�p3_1/MatMul/ReadVariableOp�p3_code/BiasAdd/ReadVariableOp�p3_code/MatMul/ReadVariableOp~
p3_1/MatMul/ReadVariableOpReadVariableOp#p3_1_matmul_readvariableop_resource*
_output_shapes

: *
dtype0s
p3_1/MatMulMatMulinputs"p3_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������|
p3_1/BiasAdd/ReadVariableOpReadVariableOp$p3_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
p3_1/BiasAddBiasAddp3_1/MatMul:product:0#p3_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������`
p3_1/SigmoidSigmoidp3_1/BiasAdd:output:0*
T0*'
_output_shapes
:���������~
p2_1/MatMul/ReadVariableOpReadVariableOp#p2_1_matmul_readvariableop_resource*
_output_shapes

: *
dtype0s
p2_1/MatMulMatMulinputs"p2_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������|
p2_1/BiasAdd/ReadVariableOpReadVariableOp$p2_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
p2_1/BiasAddBiasAddp2_1/MatMul:product:0#p2_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������`
p2_1/SigmoidSigmoidp2_1/BiasAdd:output:0*
T0*'
_output_shapes
:���������~
p1_1/MatMul/ReadVariableOpReadVariableOp#p1_1_matmul_readvariableop_resource*
_output_shapes

: *
dtype0s
p1_1/MatMulMatMulinputs"p1_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������|
p1_1/BiasAdd/ReadVariableOpReadVariableOp$p1_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
p1_1/BiasAddBiasAddp1_1/MatMul:product:0#p1_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������`
p1_1/SigmoidSigmoidp1_1/BiasAdd:output:0*
T0*'
_output_shapes
:����������
p3_code/MatMul/ReadVariableOpReadVariableOp&p3_code_matmul_readvariableop_resource*
_output_shapes

:*
dtype0�
p3_code/MatMulMatMulp3_1/Sigmoid:y:0%p3_code/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
p3_code/BiasAdd/ReadVariableOpReadVariableOp'p3_code_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
p3_code/BiasAddBiasAddp3_code/MatMul:product:0&p3_code/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������f
p3_code/SigmoidSigmoidp3_code/BiasAdd:output:0*
T0*'
_output_shapes
:����������
p2_code/MatMul/ReadVariableOpReadVariableOp&p2_code_matmul_readvariableop_resource*
_output_shapes

:*
dtype0�
p2_code/MatMulMatMulp2_1/Sigmoid:y:0%p2_code/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
p2_code/BiasAdd/ReadVariableOpReadVariableOp'p2_code_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
p2_code/BiasAddBiasAddp2_code/MatMul:product:0&p2_code/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������f
p2_code/SigmoidSigmoidp2_code/BiasAdd:output:0*
T0*'
_output_shapes
:����������
p1_code/MatMul/ReadVariableOpReadVariableOp&p1_code_matmul_readvariableop_resource*
_output_shapes

:*
dtype0�
p1_code/MatMulMatMulp1_1/Sigmoid:y:0%p1_code/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
p1_code/BiasAdd/ReadVariableOpReadVariableOp'p1_code_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
p1_code/BiasAddBiasAddp1_code/MatMul:product:0&p1_code/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������f
p1_code/SigmoidSigmoidp1_code/BiasAdd:output:0*
T0*'
_output_shapes
:���������b
IdentityIdentityp1_code/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:���������d

Identity_1Identityp2_code/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:���������d

Identity_2Identityp3_code/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^p1_1/BiasAdd/ReadVariableOp^p1_1/MatMul/ReadVariableOp^p1_code/BiasAdd/ReadVariableOp^p1_code/MatMul/ReadVariableOp^p2_1/BiasAdd/ReadVariableOp^p2_1/MatMul/ReadVariableOp^p2_code/BiasAdd/ReadVariableOp^p2_code/MatMul/ReadVariableOp^p3_1/BiasAdd/ReadVariableOp^p3_1/MatMul/ReadVariableOp^p3_code/BiasAdd/ReadVariableOp^p3_code/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:��������� : : : : : : : : : : : : 2:
p1_1/BiasAdd/ReadVariableOpp1_1/BiasAdd/ReadVariableOp28
p1_1/MatMul/ReadVariableOpp1_1/MatMul/ReadVariableOp2@
p1_code/BiasAdd/ReadVariableOpp1_code/BiasAdd/ReadVariableOp2>
p1_code/MatMul/ReadVariableOpp1_code/MatMul/ReadVariableOp2:
p2_1/BiasAdd/ReadVariableOpp2_1/BiasAdd/ReadVariableOp28
p2_1/MatMul/ReadVariableOpp2_1/MatMul/ReadVariableOp2@
p2_code/BiasAdd/ReadVariableOpp2_code/BiasAdd/ReadVariableOp2>
p2_code/MatMul/ReadVariableOpp2_code/MatMul/ReadVariableOp2:
p3_1/BiasAdd/ReadVariableOpp3_1/BiasAdd/ReadVariableOp28
p3_1/MatMul/ReadVariableOpp3_1/MatMul/ReadVariableOp2@
p3_code/BiasAdd/ReadVariableOpp3_code/BiasAdd/ReadVariableOp2>
p3_code/MatMul/ReadVariableOpp3_code/MatMul/ReadVariableOp:O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
�
'__inference_Encoder_layer_call_fn_22311
original_code
unknown: 
	unknown_0:
	unknown_1: 
	unknown_2:
	unknown_3: 
	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10:
identity

identity_1

identity_2��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCalloriginal_codeunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:���������:���������:���������*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_Encoder_layer_call_and_return_conditional_losses_22280o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:���������q

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:��������� : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:V R
'
_output_shapes
:��������� 
'
_user_specified_nameoriginal_code
�!
�
B__inference_Encoder_layer_call_and_return_conditional_losses_22280

inputs

p3_1_22187: 

p3_1_22189:

p2_1_22204: 

p2_1_22206:

p1_1_22221: 

p1_1_22223:
p3_code_22238:
p3_code_22240:
p2_code_22255:
p2_code_22257:
p1_code_22272:
p1_code_22274:
identity

identity_1

identity_2��p1_1/StatefulPartitionedCall�p1_code/StatefulPartitionedCall�p2_1/StatefulPartitionedCall�p2_code/StatefulPartitionedCall�p3_1/StatefulPartitionedCall�p3_code/StatefulPartitionedCall�
p3_1/StatefulPartitionedCallStatefulPartitionedCallinputs
p3_1_22187
p3_1_22189*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *H
fCRA
?__inference_p3_1_layer_call_and_return_conditional_losses_22186�
p2_1/StatefulPartitionedCallStatefulPartitionedCallinputs
p2_1_22204
p2_1_22206*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *H
fCRA
?__inference_p2_1_layer_call_and_return_conditional_losses_22203�
p1_1/StatefulPartitionedCallStatefulPartitionedCallinputs
p1_1_22221
p1_1_22223*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *H
fCRA
?__inference_p1_1_layer_call_and_return_conditional_losses_22220�
p3_code/StatefulPartitionedCallStatefulPartitionedCall%p3_1/StatefulPartitionedCall:output:0p3_code_22238p3_code_22240*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_p3_code_layer_call_and_return_conditional_losses_22237�
p2_code/StatefulPartitionedCallStatefulPartitionedCall%p2_1/StatefulPartitionedCall:output:0p2_code_22255p2_code_22257*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_p2_code_layer_call_and_return_conditional_losses_22254�
p1_code/StatefulPartitionedCallStatefulPartitionedCall%p1_1/StatefulPartitionedCall:output:0p1_code_22272p1_code_22274*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_p1_code_layer_call_and_return_conditional_losses_22271w
IdentityIdentity(p1_code/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������y

Identity_1Identity(p2_code/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������y

Identity_2Identity(p3_code/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^p1_1/StatefulPartitionedCall ^p1_code/StatefulPartitionedCall^p2_1/StatefulPartitionedCall ^p2_code/StatefulPartitionedCall^p3_1/StatefulPartitionedCall ^p3_code/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:��������� : : : : : : : : : : : : 2<
p1_1/StatefulPartitionedCallp1_1/StatefulPartitionedCall2B
p1_code/StatefulPartitionedCallp1_code/StatefulPartitionedCall2<
p2_1/StatefulPartitionedCallp2_1/StatefulPartitionedCall2B
p2_code/StatefulPartitionedCallp2_code/StatefulPartitionedCall2<
p3_1/StatefulPartitionedCallp3_1/StatefulPartitionedCall2B
p3_code/StatefulPartitionedCallp3_code/StatefulPartitionedCall:O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�

�
?__inference_p3_1_layer_call_and_return_conditional_losses_22186

inputs0
matmul_readvariableop_resource: -
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������V
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:���������Z
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:��������� : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�

�
B__inference_p2_code_layer_call_and_return_conditional_losses_22254

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������V
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:���������Z
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�2
�
!__inference__traced_restore_23002
file_prefix.
assignvariableop_p1_1_kernel: *
assignvariableop_1_p1_1_bias:0
assignvariableop_2_p2_1_kernel: *
assignvariableop_3_p2_1_bias:0
assignvariableop_4_p3_1_kernel: *
assignvariableop_5_p3_1_bias:3
!assignvariableop_6_p1_code_kernel:-
assignvariableop_7_p1_code_bias:3
!assignvariableop_8_p2_code_kernel:-
assignvariableop_9_p2_code_bias:4
"assignvariableop_10_p3_code_kernel:.
 assignvariableop_11_p3_code_bias:
identity_13��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_2�AssignVariableOp_3�AssignVariableOp_4�AssignVariableOp_5�AssignVariableOp_6�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*-
value$B"B B B B B B B B B B B B B �
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*H
_output_shapes6
4:::::::::::::*
dtypes
2[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOpAssignVariableOpassignvariableop_p1_1_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_1AssignVariableOpassignvariableop_1_p1_1_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_2AssignVariableOpassignvariableop_2_p2_1_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_3AssignVariableOpassignvariableop_3_p2_1_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_4AssignVariableOpassignvariableop_4_p3_1_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_5AssignVariableOpassignvariableop_5_p3_1_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_6AssignVariableOp!assignvariableop_6_p1_code_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_7AssignVariableOpassignvariableop_7_p1_code_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_8AssignVariableOp!assignvariableop_8_p2_code_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_9AssignVariableOpassignvariableop_9_p2_code_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_10AssignVariableOp"assignvariableop_10_p3_code_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_11AssignVariableOp assignvariableop_11_p3_code_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 �
Identity_12Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_13IdentityIdentity_12:output:0^NoOp_1*
T0*
_output_shapes
: �
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_13Identity_13:output:0*-
_input_shapes
: : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112(
AssignVariableOp_2AssignVariableOp_22(
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

�
B__inference_p3_code_layer_call_and_return_conditional_losses_22237

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������V
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:���������Z
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
'__inference_p2_code_layer_call_fn_22864

inputs
unknown:
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_p2_code_layer_call_and_return_conditional_losses_22254o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
'__inference_p1_code_layer_call_fn_22844

inputs
unknown:
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_p1_code_layer_call_and_return_conditional_losses_22271o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�

�
?__inference_p3_1_layer_call_and_return_conditional_losses_22835

inputs0
matmul_readvariableop_resource: -
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������V
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:���������Z
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:��������� : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�

�
B__inference_p1_code_layer_call_and_return_conditional_losses_22271

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������V
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:���������Z
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
#__inference_signature_wrapper_22613
original_code
unknown: 
	unknown_0:
	unknown_1: 
	unknown_2:
	unknown_3: 
	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10:
identity

identity_1

identity_2��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCalloriginal_codeunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:���������:���������:���������*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *)
f$R"
 __inference__wrapped_model_22168o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:���������q

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:��������� : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:V R
'
_output_shapes
:��������� 
'
_user_specified_nameoriginal_code"�	L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
G
original_code6
serving_default_original_code:0��������� ;
p1_code0
StatefulPartitionedCall:0���������;
p2_code0
StatefulPartitionedCall:1���������;
p3_code0
StatefulPartitionedCall:2���������tensorflow/serving/predict:��
�
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
layer_with_weights-5
layer-6
	variables
	trainable_variables

regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures"
_tf_keras_network
"
_tf_keras_input_layer
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
�
 	variables
!trainable_variables
"regularization_losses
#	keras_api
$__call__
*%&call_and_return_all_conditional_losses

&kernel
'bias"
_tf_keras_layer
�
(	variables
)trainable_variables
*regularization_losses
+	keras_api
,__call__
*-&call_and_return_all_conditional_losses

.kernel
/bias"
_tf_keras_layer
�
0	variables
1trainable_variables
2regularization_losses
3	keras_api
4__call__
*5&call_and_return_all_conditional_losses

6kernel
7bias"
_tf_keras_layer
�
8	variables
9trainable_variables
:regularization_losses
;	keras_api
<__call__
*=&call_and_return_all_conditional_losses

>kernel
?bias"
_tf_keras_layer
v
0
1
2
3
&4
'5
.6
/7
68
79
>10
?11"
trackable_list_wrapper
v
0
1
2
3
&4
'5
.6
/7
68
79
>10
?11"
trackable_list_wrapper
 "
trackable_list_wrapper
�
@non_trainable_variables

Alayers
Bmetrics
Clayer_regularization_losses
Dlayer_metrics
	variables
	trainable_variables

regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
Etrace_0
Ftrace_1
Gtrace_2
Htrace_32�
'__inference_Encoder_layer_call_fn_22311
'__inference_Encoder_layer_call_fn_22646
'__inference_Encoder_layer_call_fn_22679
'__inference_Encoder_layer_call_fn_22506�
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

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zEtrace_0zFtrace_1zGtrace_2zHtrace_3
�
Itrace_0
Jtrace_1
Ktrace_2
Ltrace_32�
B__inference_Encoder_layer_call_and_return_conditional_losses_22727
B__inference_Encoder_layer_call_and_return_conditional_losses_22775
B__inference_Encoder_layer_call_and_return_conditional_losses_22542
B__inference_Encoder_layer_call_and_return_conditional_losses_22578�
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

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zItrace_0zJtrace_1zKtrace_2zLtrace_3
�B�
 __inference__wrapped_model_22168original_code"�
���
FullArgSpec
args� 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
,
Mserving_default"
signature_map
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
Nnon_trainable_variables

Olayers
Pmetrics
Qlayer_regularization_losses
Rlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
Strace_02�
$__inference_p1_1_layer_call_fn_22784�
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
 zStrace_0
�
Ttrace_02�
?__inference_p1_1_layer_call_and_return_conditional_losses_22795�
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
 zTtrace_0
: 2p1_1/kernel
:2	p1_1/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
Unon_trainable_variables

Vlayers
Wmetrics
Xlayer_regularization_losses
Ylayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
Ztrace_02�
$__inference_p2_1_layer_call_fn_22804�
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
 zZtrace_0
�
[trace_02�
?__inference_p2_1_layer_call_and_return_conditional_losses_22815�
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
 z[trace_0
: 2p2_1/kernel
:2	p2_1/bias
.
&0
'1"
trackable_list_wrapper
.
&0
'1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
\non_trainable_variables

]layers
^metrics
_layer_regularization_losses
`layer_metrics
 	variables
!trainable_variables
"regularization_losses
$__call__
*%&call_and_return_all_conditional_losses
&%"call_and_return_conditional_losses"
_generic_user_object
�
atrace_02�
$__inference_p3_1_layer_call_fn_22824�
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
 zatrace_0
�
btrace_02�
?__inference_p3_1_layer_call_and_return_conditional_losses_22835�
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
 zbtrace_0
: 2p3_1/kernel
:2	p3_1/bias
.
.0
/1"
trackable_list_wrapper
.
.0
/1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
cnon_trainable_variables

dlayers
emetrics
flayer_regularization_losses
glayer_metrics
(	variables
)trainable_variables
*regularization_losses
,__call__
*-&call_and_return_all_conditional_losses
&-"call_and_return_conditional_losses"
_generic_user_object
�
htrace_02�
'__inference_p1_code_layer_call_fn_22844�
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
 zhtrace_0
�
itrace_02�
B__inference_p1_code_layer_call_and_return_conditional_losses_22855�
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
 zitrace_0
 :2p1_code/kernel
:2p1_code/bias
.
60
71"
trackable_list_wrapper
.
60
71"
trackable_list_wrapper
 "
trackable_list_wrapper
�
jnon_trainable_variables

klayers
lmetrics
mlayer_regularization_losses
nlayer_metrics
0	variables
1trainable_variables
2regularization_losses
4__call__
*5&call_and_return_all_conditional_losses
&5"call_and_return_conditional_losses"
_generic_user_object
�
otrace_02�
'__inference_p2_code_layer_call_fn_22864�
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
 zotrace_0
�
ptrace_02�
B__inference_p2_code_layer_call_and_return_conditional_losses_22875�
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
 zptrace_0
 :2p2_code/kernel
:2p2_code/bias
.
>0
?1"
trackable_list_wrapper
.
>0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
qnon_trainable_variables

rlayers
smetrics
tlayer_regularization_losses
ulayer_metrics
8	variables
9trainable_variables
:regularization_losses
<__call__
*=&call_and_return_all_conditional_losses
&="call_and_return_conditional_losses"
_generic_user_object
�
vtrace_02�
'__inference_p3_code_layer_call_fn_22884�
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
 zvtrace_0
�
wtrace_02�
B__inference_p3_code_layer_call_and_return_conditional_losses_22895�
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
 zwtrace_0
 :2p3_code/kernel
:2p3_code/bias
 "
trackable_list_wrapper
Q
0
1
2
3
4
5
6"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
'__inference_Encoder_layer_call_fn_22311original_code"�
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

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
'__inference_Encoder_layer_call_fn_22646inputs"�
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

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
'__inference_Encoder_layer_call_fn_22679inputs"�
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

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
'__inference_Encoder_layer_call_fn_22506original_code"�
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

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
B__inference_Encoder_layer_call_and_return_conditional_losses_22727inputs"�
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

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
B__inference_Encoder_layer_call_and_return_conditional_losses_22775inputs"�
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

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
B__inference_Encoder_layer_call_and_return_conditional_losses_22542original_code"�
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

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
B__inference_Encoder_layer_call_and_return_conditional_losses_22578original_code"�
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

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
#__inference_signature_wrapper_22613original_code"�
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
 
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
�B�
$__inference_p1_1_layer_call_fn_22784inputs"�
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
?__inference_p1_1_layer_call_and_return_conditional_losses_22795inputs"�
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
�B�
$__inference_p2_1_layer_call_fn_22804inputs"�
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
?__inference_p2_1_layer_call_and_return_conditional_losses_22815inputs"�
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
�B�
$__inference_p3_1_layer_call_fn_22824inputs"�
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
?__inference_p3_1_layer_call_and_return_conditional_losses_22835inputs"�
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
�B�
'__inference_p1_code_layer_call_fn_22844inputs"�
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
B__inference_p1_code_layer_call_and_return_conditional_losses_22855inputs"�
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
�B�
'__inference_p2_code_layer_call_fn_22864inputs"�
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
B__inference_p2_code_layer_call_and_return_conditional_losses_22875inputs"�
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
�B�
'__inference_p3_code_layer_call_fn_22884inputs"�
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
B__inference_p3_code_layer_call_and_return_conditional_losses_22895inputs"�
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
 �
B__inference_Encoder_layer_call_and_return_conditional_losses_22542�&'>?67./>�;
4�1
'�$
original_code��������� 
p 

 
� "j�g
`�]
�
0/0���������
�
0/1���������
�
0/2���������
� �
B__inference_Encoder_layer_call_and_return_conditional_losses_22578�&'>?67./>�;
4�1
'�$
original_code��������� 
p

 
� "j�g
`�]
�
0/0���������
�
0/1���������
�
0/2���������
� �
B__inference_Encoder_layer_call_and_return_conditional_losses_22727�&'>?67./7�4
-�*
 �
inputs��������� 
p 

 
� "j�g
`�]
�
0/0���������
�
0/1���������
�
0/2���������
� �
B__inference_Encoder_layer_call_and_return_conditional_losses_22775�&'>?67./7�4
-�*
 �
inputs��������� 
p

 
� "j�g
`�]
�
0/0���������
�
0/1���������
�
0/2���������
� �
'__inference_Encoder_layer_call_fn_22311�&'>?67./>�;
4�1
'�$
original_code��������� 
p 

 
� "Z�W
�
0���������
�
1���������
�
2����������
'__inference_Encoder_layer_call_fn_22506�&'>?67./>�;
4�1
'�$
original_code��������� 
p

 
� "Z�W
�
0���������
�
1���������
�
2����������
'__inference_Encoder_layer_call_fn_22646�&'>?67./7�4
-�*
 �
inputs��������� 
p 

 
� "Z�W
�
0���������
�
1���������
�
2����������
'__inference_Encoder_layer_call_fn_22679�&'>?67./7�4
-�*
 �
inputs��������� 
p

 
� "Z�W
�
0���������
�
1���������
�
2����������
 __inference__wrapped_model_22168�&'>?67./6�3
,�)
'�$
original_code��������� 
� "���
,
p1_code!�
p1_code���������
,
p2_code!�
p2_code���������
,
p3_code!�
p3_code����������
?__inference_p1_1_layer_call_and_return_conditional_losses_22795\/�,
%�"
 �
inputs��������� 
� "%�"
�
0���������
� w
$__inference_p1_1_layer_call_fn_22784O/�,
%�"
 �
inputs��������� 
� "�����������
B__inference_p1_code_layer_call_and_return_conditional_losses_22855\.//�,
%�"
 �
inputs���������
� "%�"
�
0���������
� z
'__inference_p1_code_layer_call_fn_22844O.//�,
%�"
 �
inputs���������
� "�����������
?__inference_p2_1_layer_call_and_return_conditional_losses_22815\/�,
%�"
 �
inputs��������� 
� "%�"
�
0���������
� w
$__inference_p2_1_layer_call_fn_22804O/�,
%�"
 �
inputs��������� 
� "�����������
B__inference_p2_code_layer_call_and_return_conditional_losses_22875\67/�,
%�"
 �
inputs���������
� "%�"
�
0���������
� z
'__inference_p2_code_layer_call_fn_22864O67/�,
%�"
 �
inputs���������
� "�����������
?__inference_p3_1_layer_call_and_return_conditional_losses_22835\&'/�,
%�"
 �
inputs��������� 
� "%�"
�
0���������
� w
$__inference_p3_1_layer_call_fn_22824O&'/�,
%�"
 �
inputs��������� 
� "�����������
B__inference_p3_code_layer_call_and_return_conditional_losses_22895\>?/�,
%�"
 �
inputs���������
� "%�"
�
0���������
� z
'__inference_p3_code_layer_call_fn_22884O>?/�,
%�"
 �
inputs���������
� "�����������
#__inference_signature_wrapper_22613�&'>?67./G�D
� 
=�:
8
original_code'�$
original_code��������� "���
,
p1_code!�
p1_code���������
,
p2_code!�
p2_code���������
,
p3_code!�
p3_code���������