‘0
Ρ£
8
Const
output"dtype"
valuetensor"
dtypetype

NoOp
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype
Ύ
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
executor_typestring 

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.3.02v2.3.0-rc2-23-gb36436b0878ΘΗ%
~
conv_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv_1/kernel
w
!conv_1/kernel/Read/ReadVariableOpReadVariableOpconv_1/kernel*&
_output_shapes
:*
dtype0
n
conv_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv_1/bias
g
conv_1/bias/Read/ReadVariableOpReadVariableOpconv_1/bias*
_output_shapes
:*
dtype0

batch_normalization/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_namebatch_normalization/gamma

-batch_normalization/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization/gamma*
_output_shapes
:*
dtype0

batch_normalization/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_namebatch_normalization/beta

,batch_normalization/beta/Read/ReadVariableOpReadVariableOpbatch_normalization/beta*
_output_shapes
:*
dtype0

batch_normalization/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*0
shared_name!batch_normalization/moving_mean

3batch_normalization/moving_mean/Read/ReadVariableOpReadVariableOpbatch_normalization/moving_mean*
_output_shapes
:*
dtype0

#batch_normalization/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#batch_normalization/moving_variance

7batch_normalization/moving_variance/Read/ReadVariableOpReadVariableOp#batch_normalization/moving_variance*
_output_shapes
:*
dtype0
~
conv_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv_2/kernel
w
!conv_2/kernel/Read/ReadVariableOpReadVariableOpconv_2/kernel*&
_output_shapes
: *
dtype0
n
conv_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv_2/bias
g
conv_2/bias/Read/ReadVariableOpReadVariableOpconv_2/bias*
_output_shapes
: *
dtype0

batch_normalization_1/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *,
shared_namebatch_normalization_1/gamma

/batch_normalization_1/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_1/gamma*
_output_shapes
: *
dtype0

batch_normalization_1/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *+
shared_namebatch_normalization_1/beta

.batch_normalization_1/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_1/beta*
_output_shapes
: *
dtype0

!batch_normalization_1/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape: *2
shared_name#!batch_normalization_1/moving_mean

5batch_normalization_1/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_1/moving_mean*
_output_shapes
: *
dtype0
’
%batch_normalization_1/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape: *6
shared_name'%batch_normalization_1/moving_variance

9batch_normalization_1/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_1/moving_variance*
_output_shapes
: *
dtype0
~
conv_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*
shared_nameconv_3/kernel
w
!conv_3/kernel/Read/ReadVariableOpReadVariableOpconv_3/kernel*&
_output_shapes
: @*
dtype0
n
conv_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv_3/bias
g
conv_3/bias/Read/ReadVariableOpReadVariableOpconv_3/bias*
_output_shapes
:@*
dtype0

batch_normalization_2/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_namebatch_normalization_2/gamma

/batch_normalization_2/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_2/gamma*
_output_shapes
:@*
dtype0

batch_normalization_2/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*+
shared_namebatch_normalization_2/beta

.batch_normalization_2/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_2/beta*
_output_shapes
:@*
dtype0

!batch_normalization_2/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*2
shared_name#!batch_normalization_2/moving_mean

5batch_normalization_2/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_2/moving_mean*
_output_shapes
:@*
dtype0
’
%batch_normalization_2/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*6
shared_name'%batch_normalization_2/moving_variance

9batch_normalization_2/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_2/moving_variance*
_output_shapes
:@*
dtype0

conv_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv_4/kernel
x
!conv_4/kernel/Read/ReadVariableOpReadVariableOpconv_4/kernel*'
_output_shapes
:@*
dtype0
o
conv_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv_4/bias
h
conv_4/bias/Read/ReadVariableOpReadVariableOpconv_4/bias*
_output_shapes	
:*
dtype0

batch_normalization_3/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatch_normalization_3/gamma

/batch_normalization_3/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_3/gamma*
_output_shapes	
:*
dtype0

batch_normalization_3/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_namebatch_normalization_3/beta

.batch_normalization_3/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_3/beta*
_output_shapes	
:*
dtype0

!batch_normalization_3/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!batch_normalization_3/moving_mean

5batch_normalization_3/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_3/moving_mean*
_output_shapes	
:*
dtype0
£
%batch_normalization_3/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*6
shared_name'%batch_normalization_3/moving_variance

9batch_normalization_3/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_3/moving_variance*
_output_shapes	
:*
dtype0

conv_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv_5/kernel
y
!conv_5/kernel/Read/ReadVariableOpReadVariableOpconv_5/kernel*(
_output_shapes
:*
dtype0
o
conv_5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv_5/bias
h
conv_5/bias/Read/ReadVariableOpReadVariableOpconv_5/bias*
_output_shapes	
:*
dtype0

batch_normalization_4/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatch_normalization_4/gamma

/batch_normalization_4/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_4/gamma*
_output_shapes	
:*
dtype0

batch_normalization_4/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_namebatch_normalization_4/beta

.batch_normalization_4/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_4/beta*
_output_shapes	
:*
dtype0

!batch_normalization_4/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!batch_normalization_4/moving_mean

5batch_normalization_4/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_4/moving_mean*
_output_shapes	
:*
dtype0
£
%batch_normalization_4/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*6
shared_name'%batch_normalization_4/moving_variance

9batch_normalization_4/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_4/moving_variance*
_output_shapes	
:*
dtype0

upsample_6/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameupsample_6/kernel

%upsample_6/kernel/Read/ReadVariableOpReadVariableOpupsample_6/kernel*(
_output_shapes
:*
dtype0
w
upsample_6/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameupsample_6/bias
p
#upsample_6/bias/Read/ReadVariableOpReadVariableOpupsample_6/bias*
_output_shapes	
:*
dtype0

conv_6/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv_6/kernel
y
!conv_6/kernel/Read/ReadVariableOpReadVariableOpconv_6/kernel*(
_output_shapes
:*
dtype0
o
conv_6/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv_6/bias
h
conv_6/bias/Read/ReadVariableOpReadVariableOpconv_6/bias*
_output_shapes	
:*
dtype0

batch_normalization_5/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatch_normalization_5/gamma

/batch_normalization_5/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_5/gamma*
_output_shapes	
:*
dtype0

batch_normalization_5/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_namebatch_normalization_5/beta

.batch_normalization_5/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_5/beta*
_output_shapes	
:*
dtype0

!batch_normalization_5/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!batch_normalization_5/moving_mean

5batch_normalization_5/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_5/moving_mean*
_output_shapes	
:*
dtype0
£
%batch_normalization_5/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*6
shared_name'%batch_normalization_5/moving_variance

9batch_normalization_5/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_5/moving_variance*
_output_shapes	
:*
dtype0

upsample_7/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*"
shared_nameupsample_7/kernel

%upsample_7/kernel/Read/ReadVariableOpReadVariableOpupsample_7/kernel*'
_output_shapes
:@*
dtype0
v
upsample_7/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@* 
shared_nameupsample_7/bias
o
#upsample_7/bias/Read/ReadVariableOpReadVariableOpupsample_7/bias*
_output_shapes
:@*
dtype0

conv_7/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv_7/kernel
x
!conv_7/kernel/Read/ReadVariableOpReadVariableOpconv_7/kernel*'
_output_shapes
:@*
dtype0
n
conv_7/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv_7/bias
g
conv_7/bias/Read/ReadVariableOpReadVariableOpconv_7/bias*
_output_shapes
:@*
dtype0

batch_normalization_6/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_namebatch_normalization_6/gamma

/batch_normalization_6/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_6/gamma*
_output_shapes
:@*
dtype0

batch_normalization_6/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*+
shared_namebatch_normalization_6/beta

.batch_normalization_6/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_6/beta*
_output_shapes
:@*
dtype0

!batch_normalization_6/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*2
shared_name#!batch_normalization_6/moving_mean

5batch_normalization_6/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_6/moving_mean*
_output_shapes
:@*
dtype0
’
%batch_normalization_6/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*6
shared_name'%batch_normalization_6/moving_variance

9batch_normalization_6/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_6/moving_variance*
_output_shapes
:@*
dtype0

upsample_8/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*"
shared_nameupsample_8/kernel

%upsample_8/kernel/Read/ReadVariableOpReadVariableOpupsample_8/kernel*&
_output_shapes
: @*
dtype0
v
upsample_8/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: * 
shared_nameupsample_8/bias
o
#upsample_8/bias/Read/ReadVariableOpReadVariableOpupsample_8/bias*
_output_shapes
: *
dtype0
~
conv_8/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@ *
shared_nameconv_8/kernel
w
!conv_8/kernel/Read/ReadVariableOpReadVariableOpconv_8/kernel*&
_output_shapes
:@ *
dtype0
n
conv_8/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv_8/bias
g
conv_8/bias/Read/ReadVariableOpReadVariableOpconv_8/bias*
_output_shapes
: *
dtype0

batch_normalization_7/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *,
shared_namebatch_normalization_7/gamma

/batch_normalization_7/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_7/gamma*
_output_shapes
: *
dtype0

batch_normalization_7/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *+
shared_namebatch_normalization_7/beta

.batch_normalization_7/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_7/beta*
_output_shapes
: *
dtype0

!batch_normalization_7/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape: *2
shared_name#!batch_normalization_7/moving_mean

5batch_normalization_7/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_7/moving_mean*
_output_shapes
: *
dtype0
’
%batch_normalization_7/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape: *6
shared_name'%batch_normalization_7/moving_variance

9batch_normalization_7/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_7/moving_variance*
_output_shapes
: *
dtype0

upsample_9/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *"
shared_nameupsample_9/kernel

%upsample_9/kernel/Read/ReadVariableOpReadVariableOpupsample_9/kernel*&
_output_shapes
: *
dtype0
v
upsample_9/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameupsample_9/bias
o
#upsample_9/bias/Read/ReadVariableOpReadVariableOpupsample_9/bias*
_output_shapes
:*
dtype0
~
conv_9/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv_9/kernel
w
!conv_9/kernel/Read/ReadVariableOpReadVariableOpconv_9/kernel*&
_output_shapes
: *
dtype0
n
conv_9/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv_9/bias
g
conv_9/bias/Read/ReadVariableOpReadVariableOpconv_9/bias*
_output_shapes
:*
dtype0

batch_normalization_8/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatch_normalization_8/gamma

/batch_normalization_8/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_8/gamma*
_output_shapes
:*
dtype0

batch_normalization_8/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_namebatch_normalization_8/beta

.batch_normalization_8/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_8/beta*
_output_shapes
:*
dtype0

!batch_normalization_8/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!batch_normalization_8/moving_mean

5batch_normalization_8/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_8/moving_mean*
_output_shapes
:*
dtype0
’
%batch_normalization_8/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*6
shared_name'%batch_normalization_8/moving_variance

9batch_normalization_8/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_8/moving_variance*
_output_shapes
:*
dtype0
|
final/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namefinal/kernel
u
 final/kernel/Read/ReadVariableOpReadVariableOpfinal/kernel*&
_output_shapes
:*
dtype0
l

final/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name
final/bias
e
final/bias/Read/ReadVariableOpReadVariableOp
final/bias*
_output_shapes
:*
dtype0

NoOpNoOp
Έ
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*Κ·
valueΏ·B»· B³·


layer-0
layer-1
layer-2
layer_with_weights-0
layer-3
layer_with_weights-1
layer-4
layer-5
layer-6
layer_with_weights-2
layer-7
	layer_with_weights-3
	layer-8

layer-9
layer-10
layer_with_weights-4
layer-11
layer_with_weights-5
layer-12
layer-13
layer-14
layer_with_weights-6
layer-15
layer_with_weights-7
layer-16
layer-17
layer-18
layer_with_weights-8
layer-19
layer_with_weights-9
layer-20
layer-21
layer_with_weights-10
layer-22
layer-23
layer_with_weights-11
layer-24
layer_with_weights-12
layer-25
layer-26
layer_with_weights-13
layer-27
layer-28
layer_with_weights-14
layer-29
layer_with_weights-15
layer-30
 layer-31
!layer_with_weights-16
!layer-32
"layer-33
#layer_with_weights-17
#layer-34
$layer_with_weights-18
$layer-35
%layer-36
&layer_with_weights-19
&layer-37
'layer-38
(layer_with_weights-20
(layer-39
)layer_with_weights-21
)layer-40
*layer-41
+layer_with_weights-22
+layer-42
,regularization_losses
-	variables
.trainable_variables
/	keras_api
0
signatures
 
R
1regularization_losses
2	variables
3trainable_variables
4	keras_api
R
5regularization_losses
6	variables
7trainable_variables
8	keras_api
h

9kernel
:bias
;regularization_losses
<	variables
=trainable_variables
>	keras_api

?axis
	@gamma
Abeta
Bmoving_mean
Cmoving_variance
Dregularization_losses
E	variables
Ftrainable_variables
G	keras_api
R
Hregularization_losses
I	variables
Jtrainable_variables
K	keras_api
R
Lregularization_losses
M	variables
Ntrainable_variables
O	keras_api
h

Pkernel
Qbias
Rregularization_losses
S	variables
Ttrainable_variables
U	keras_api

Vaxis
	Wgamma
Xbeta
Ymoving_mean
Zmoving_variance
[regularization_losses
\	variables
]trainable_variables
^	keras_api
R
_regularization_losses
`	variables
atrainable_variables
b	keras_api
R
cregularization_losses
d	variables
etrainable_variables
f	keras_api
h

gkernel
hbias
iregularization_losses
j	variables
ktrainable_variables
l	keras_api

maxis
	ngamma
obeta
pmoving_mean
qmoving_variance
rregularization_losses
s	variables
ttrainable_variables
u	keras_api
R
vregularization_losses
w	variables
xtrainable_variables
y	keras_api
R
zregularization_losses
{	variables
|trainable_variables
}	keras_api
l

~kernel
bias
regularization_losses
	variables
trainable_variables
	keras_api
 
	axis

gamma
	beta
moving_mean
moving_variance
regularization_losses
	variables
trainable_variables
	keras_api
V
regularization_losses
	variables
trainable_variables
	keras_api
V
regularization_losses
	variables
trainable_variables
	keras_api
n
kernel
	bias
regularization_losses
	variables
trainable_variables
	keras_api
 
	axis

gamma
	beta
moving_mean
moving_variance
 regularization_losses
‘	variables
’trainable_variables
£	keras_api
V
€regularization_losses
₯	variables
¦trainable_variables
§	keras_api
n
¨kernel
	©bias
ͺregularization_losses
«	variables
¬trainable_variables
­	keras_api
V
?regularization_losses
―	variables
°trainable_variables
±	keras_api
n
²kernel
	³bias
΄regularization_losses
΅	variables
Άtrainable_variables
·	keras_api
 
	Έaxis

Ήgamma
	Ίbeta
»moving_mean
Όmoving_variance
½regularization_losses
Ύ	variables
Ώtrainable_variables
ΐ	keras_api
V
Αregularization_losses
Β	variables
Γtrainable_variables
Δ	keras_api
n
Εkernel
	Ζbias
Ηregularization_losses
Θ	variables
Ιtrainable_variables
Κ	keras_api
V
Λregularization_losses
Μ	variables
Νtrainable_variables
Ξ	keras_api
n
Οkernel
	Πbias
Ρregularization_losses
?	variables
Σtrainable_variables
Τ	keras_api
 
	Υaxis

Φgamma
	Χbeta
Ψmoving_mean
Ωmoving_variance
Ϊregularization_losses
Ϋ	variables
άtrainable_variables
έ	keras_api
V
ήregularization_losses
ί	variables
ΰtrainable_variables
α	keras_api
n
βkernel
	γbias
δregularization_losses
ε	variables
ζtrainable_variables
η	keras_api
V
θregularization_losses
ι	variables
κtrainable_variables
λ	keras_api
n
μkernel
	νbias
ξregularization_losses
ο	variables
πtrainable_variables
ρ	keras_api
 
	ςaxis

σgamma
	τbeta
υmoving_mean
φmoving_variance
χregularization_losses
ψ	variables
ωtrainable_variables
ϊ	keras_api
V
ϋregularization_losses
ό	variables
ύtrainable_variables
ώ	keras_api
n
?kernel
	bias
regularization_losses
	variables
trainable_variables
	keras_api
V
regularization_losses
	variables
trainable_variables
	keras_api
n
kernel
	bias
regularization_losses
	variables
trainable_variables
	keras_api
 
	axis

gamma
	beta
moving_mean
moving_variance
regularization_losses
	variables
trainable_variables
	keras_api
V
regularization_losses
	variables
trainable_variables
	keras_api
n
kernel
	bias
regularization_losses
	variables
 trainable_variables
‘	keras_api
 
’
90
:1
@2
A3
B4
C5
P6
Q7
W8
X9
Y10
Z11
g12
h13
n14
o15
p16
q17
~18
19
20
21
22
23
24
25
26
27
28
29
¨30
©31
²32
³33
Ή34
Ί35
»36
Ό37
Ε38
Ζ39
Ο40
Π41
Φ42
Χ43
Ψ44
Ω45
β46
γ47
μ48
ν49
σ50
τ51
υ52
φ53
?54
55
56
57
58
59
60
61
62
63

90
:1
@2
A3
P4
Q5
W6
X7
g8
h9
n10
o11
~12
13
14
15
16
17
18
19
¨20
©21
²22
³23
Ή24
Ί25
Ε26
Ζ27
Ο28
Π29
Φ30
Χ31
β32
γ33
μ34
ν35
σ36
τ37
?38
39
40
41
42
43
44
45
²
,regularization_losses
’non_trainable_variables
-	variables
£metrics
 €layer_regularization_losses
₯layer_metrics
¦layers
.trainable_variables
 
 
 
 
²
1regularization_losses
§non_trainable_variables
2	variables
¨metrics
 ©layer_regularization_losses
ͺlayer_metrics
«layers
3trainable_variables
 
 
 
²
5regularization_losses
¬non_trainable_variables
6	variables
­metrics
 ?layer_regularization_losses
―layer_metrics
°layers
7trainable_variables
YW
VARIABLE_VALUEconv_1/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEconv_1/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE
 

90
:1

90
:1
²
;regularization_losses
±non_trainable_variables
<	variables
²metrics
 ³layer_regularization_losses
΄layer_metrics
΅layers
=trainable_variables
 
db
VARIABLE_VALUEbatch_normalization/gamma5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUEbatch_normalization/beta4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUE
pn
VARIABLE_VALUEbatch_normalization/moving_mean;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUE#batch_normalization/moving_variance?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 

@0
A1
B2
C3

@0
A1
²
Dregularization_losses
Άnon_trainable_variables
E	variables
·metrics
 Έlayer_regularization_losses
Ήlayer_metrics
Ίlayers
Ftrainable_variables
 
 
 
²
Hregularization_losses
»non_trainable_variables
I	variables
Όmetrics
 ½layer_regularization_losses
Ύlayer_metrics
Ώlayers
Jtrainable_variables
 
 
 
²
Lregularization_losses
ΐnon_trainable_variables
M	variables
Αmetrics
 Βlayer_regularization_losses
Γlayer_metrics
Δlayers
Ntrainable_variables
YW
VARIABLE_VALUEconv_2/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEconv_2/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE
 

P0
Q1

P0
Q1
²
Rregularization_losses
Εnon_trainable_variables
S	variables
Ζmetrics
 Ηlayer_regularization_losses
Θlayer_metrics
Ιlayers
Ttrainable_variables
 
fd
VARIABLE_VALUEbatch_normalization_1/gamma5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUE
db
VARIABLE_VALUEbatch_normalization_1/beta4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUE!batch_normalization_1/moving_mean;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE%batch_normalization_1/moving_variance?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 

W0
X1
Y2
Z3

W0
X1
²
[regularization_losses
Κnon_trainable_variables
\	variables
Λmetrics
 Μlayer_regularization_losses
Νlayer_metrics
Ξlayers
]trainable_variables
 
 
 
²
_regularization_losses
Οnon_trainable_variables
`	variables
Πmetrics
 Ρlayer_regularization_losses
?layer_metrics
Σlayers
atrainable_variables
 
 
 
²
cregularization_losses
Τnon_trainable_variables
d	variables
Υmetrics
 Φlayer_regularization_losses
Χlayer_metrics
Ψlayers
etrainable_variables
YW
VARIABLE_VALUEconv_3/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEconv_3/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE
 

g0
h1

g0
h1
²
iregularization_losses
Ωnon_trainable_variables
j	variables
Ϊmetrics
 Ϋlayer_regularization_losses
άlayer_metrics
έlayers
ktrainable_variables
 
fd
VARIABLE_VALUEbatch_normalization_2/gamma5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUE
db
VARIABLE_VALUEbatch_normalization_2/beta4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUE!batch_normalization_2/moving_mean;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE%batch_normalization_2/moving_variance?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 

n0
o1
p2
q3

n0
o1
²
rregularization_losses
ήnon_trainable_variables
s	variables
ίmetrics
 ΰlayer_regularization_losses
αlayer_metrics
βlayers
ttrainable_variables
 
 
 
²
vregularization_losses
γnon_trainable_variables
w	variables
δmetrics
 εlayer_regularization_losses
ζlayer_metrics
ηlayers
xtrainable_variables
 
 
 
²
zregularization_losses
θnon_trainable_variables
{	variables
ιmetrics
 κlayer_regularization_losses
λlayer_metrics
μlayers
|trainable_variables
YW
VARIABLE_VALUEconv_4/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEconv_4/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE
 

~0
1

~0
1
΅
regularization_losses
νnon_trainable_variables
	variables
ξmetrics
 οlayer_regularization_losses
πlayer_metrics
ρlayers
trainable_variables
 
fd
VARIABLE_VALUEbatch_normalization_3/gamma5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUE
db
VARIABLE_VALUEbatch_normalization_3/beta4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUE!batch_normalization_3/moving_mean;layer_with_weights-7/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE%batch_normalization_3/moving_variance?layer_with_weights-7/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 
 
0
1
2
3

0
1
΅
regularization_losses
ςnon_trainable_variables
	variables
σmetrics
 τlayer_regularization_losses
υlayer_metrics
φlayers
trainable_variables
 
 
 
΅
regularization_losses
χnon_trainable_variables
	variables
ψmetrics
 ωlayer_regularization_losses
ϊlayer_metrics
ϋlayers
trainable_variables
 
 
 
΅
regularization_losses
όnon_trainable_variables
	variables
ύmetrics
 ώlayer_regularization_losses
?layer_metrics
layers
trainable_variables
YW
VARIABLE_VALUEconv_5/kernel6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEconv_5/bias4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
΅
regularization_losses
non_trainable_variables
	variables
metrics
 layer_regularization_losses
layer_metrics
layers
trainable_variables
 
fd
VARIABLE_VALUEbatch_normalization_4/gamma5layer_with_weights-9/gamma/.ATTRIBUTES/VARIABLE_VALUE
db
VARIABLE_VALUEbatch_normalization_4/beta4layer_with_weights-9/beta/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUE!batch_normalization_4/moving_mean;layer_with_weights-9/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE%batch_normalization_4/moving_variance?layer_with_weights-9/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 
 
0
1
2
3

0
1
΅
 regularization_losses
non_trainable_variables
‘	variables
metrics
 layer_regularization_losses
layer_metrics
layers
’trainable_variables
 
 
 
΅
€regularization_losses
non_trainable_variables
₯	variables
metrics
 layer_regularization_losses
layer_metrics
layers
¦trainable_variables
^\
VARIABLE_VALUEupsample_6/kernel7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEupsample_6/bias5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUE
 

¨0
©1

¨0
©1
΅
ͺregularization_losses
non_trainable_variables
«	variables
metrics
 layer_regularization_losses
layer_metrics
layers
¬trainable_variables
 
 
 
΅
?regularization_losses
non_trainable_variables
―	variables
metrics
 layer_regularization_losses
layer_metrics
layers
°trainable_variables
ZX
VARIABLE_VALUEconv_6/kernel7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEconv_6/bias5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUE
 

²0
³1

²0
³1
΅
΄regularization_losses
non_trainable_variables
΅	variables
metrics
 layer_regularization_losses
layer_metrics
layers
Άtrainable_variables
 
ge
VARIABLE_VALUEbatch_normalization_5/gamma6layer_with_weights-12/gamma/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEbatch_normalization_5/beta5layer_with_weights-12/beta/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE!batch_normalization_5/moving_mean<layer_with_weights-12/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE%batch_normalization_5/moving_variance@layer_with_weights-12/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 
 
Ή0
Ί1
»2
Ό3

Ή0
Ί1
΅
½regularization_losses
non_trainable_variables
Ύ	variables
 metrics
 ‘layer_regularization_losses
’layer_metrics
£layers
Ώtrainable_variables
 
 
 
΅
Αregularization_losses
€non_trainable_variables
Β	variables
₯metrics
 ¦layer_regularization_losses
§layer_metrics
¨layers
Γtrainable_variables
^\
VARIABLE_VALUEupsample_7/kernel7layer_with_weights-13/kernel/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEupsample_7/bias5layer_with_weights-13/bias/.ATTRIBUTES/VARIABLE_VALUE
 

Ε0
Ζ1

Ε0
Ζ1
΅
Ηregularization_losses
©non_trainable_variables
Θ	variables
ͺmetrics
 «layer_regularization_losses
¬layer_metrics
­layers
Ιtrainable_variables
 
 
 
΅
Λregularization_losses
?non_trainable_variables
Μ	variables
―metrics
 °layer_regularization_losses
±layer_metrics
²layers
Νtrainable_variables
ZX
VARIABLE_VALUEconv_7/kernel7layer_with_weights-14/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEconv_7/bias5layer_with_weights-14/bias/.ATTRIBUTES/VARIABLE_VALUE
 

Ο0
Π1

Ο0
Π1
΅
Ρregularization_losses
³non_trainable_variables
?	variables
΄metrics
 ΅layer_regularization_losses
Άlayer_metrics
·layers
Σtrainable_variables
 
ge
VARIABLE_VALUEbatch_normalization_6/gamma6layer_with_weights-15/gamma/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEbatch_normalization_6/beta5layer_with_weights-15/beta/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE!batch_normalization_6/moving_mean<layer_with_weights-15/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE%batch_normalization_6/moving_variance@layer_with_weights-15/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 
 
Φ0
Χ1
Ψ2
Ω3

Φ0
Χ1
΅
Ϊregularization_losses
Έnon_trainable_variables
Ϋ	variables
Ήmetrics
 Ίlayer_regularization_losses
»layer_metrics
Όlayers
άtrainable_variables
 
 
 
΅
ήregularization_losses
½non_trainable_variables
ί	variables
Ύmetrics
 Ώlayer_regularization_losses
ΐlayer_metrics
Αlayers
ΰtrainable_variables
^\
VARIABLE_VALUEupsample_8/kernel7layer_with_weights-16/kernel/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEupsample_8/bias5layer_with_weights-16/bias/.ATTRIBUTES/VARIABLE_VALUE
 

β0
γ1

β0
γ1
΅
δregularization_losses
Βnon_trainable_variables
ε	variables
Γmetrics
 Δlayer_regularization_losses
Εlayer_metrics
Ζlayers
ζtrainable_variables
 
 
 
΅
θregularization_losses
Ηnon_trainable_variables
ι	variables
Θmetrics
 Ιlayer_regularization_losses
Κlayer_metrics
Λlayers
κtrainable_variables
ZX
VARIABLE_VALUEconv_8/kernel7layer_with_weights-17/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEconv_8/bias5layer_with_weights-17/bias/.ATTRIBUTES/VARIABLE_VALUE
 

μ0
ν1

μ0
ν1
΅
ξregularization_losses
Μnon_trainable_variables
ο	variables
Νmetrics
 Ξlayer_regularization_losses
Οlayer_metrics
Πlayers
πtrainable_variables
 
ge
VARIABLE_VALUEbatch_normalization_7/gamma6layer_with_weights-18/gamma/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEbatch_normalization_7/beta5layer_with_weights-18/beta/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE!batch_normalization_7/moving_mean<layer_with_weights-18/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE%batch_normalization_7/moving_variance@layer_with_weights-18/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 
 
σ0
τ1
υ2
φ3

σ0
τ1
΅
χregularization_losses
Ρnon_trainable_variables
ψ	variables
?metrics
 Σlayer_regularization_losses
Τlayer_metrics
Υlayers
ωtrainable_variables
 
 
 
΅
ϋregularization_losses
Φnon_trainable_variables
ό	variables
Χmetrics
 Ψlayer_regularization_losses
Ωlayer_metrics
Ϊlayers
ύtrainable_variables
^\
VARIABLE_VALUEupsample_9/kernel7layer_with_weights-19/kernel/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEupsample_9/bias5layer_with_weights-19/bias/.ATTRIBUTES/VARIABLE_VALUE
 

?0
1

?0
1
΅
regularization_losses
Ϋnon_trainable_variables
	variables
άmetrics
 έlayer_regularization_losses
ήlayer_metrics
ίlayers
trainable_variables
 
 
 
΅
regularization_losses
ΰnon_trainable_variables
	variables
αmetrics
 βlayer_regularization_losses
γlayer_metrics
δlayers
trainable_variables
ZX
VARIABLE_VALUEconv_9/kernel7layer_with_weights-20/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEconv_9/bias5layer_with_weights-20/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
΅
regularization_losses
εnon_trainable_variables
	variables
ζmetrics
 ηlayer_regularization_losses
θlayer_metrics
ιlayers
trainable_variables
 
ge
VARIABLE_VALUEbatch_normalization_8/gamma6layer_with_weights-21/gamma/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEbatch_normalization_8/beta5layer_with_weights-21/beta/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE!batch_normalization_8/moving_mean<layer_with_weights-21/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE%batch_normalization_8/moving_variance@layer_with_weights-21/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 
 
0
1
2
3

0
1
΅
regularization_losses
κnon_trainable_variables
	variables
λmetrics
 μlayer_regularization_losses
νlayer_metrics
ξlayers
trainable_variables
 
 
 
΅
regularization_losses
οnon_trainable_variables
	variables
πmetrics
 ρlayer_regularization_losses
ςlayer_metrics
σlayers
trainable_variables
YW
VARIABLE_VALUEfinal/kernel7layer_with_weights-22/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUE
final/bias5layer_with_weights-22/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
΅
regularization_losses
τnon_trainable_variables
	variables
υmetrics
 φlayer_regularization_losses
χlayer_metrics
ψlayers
 trainable_variables

B0
C1
Y2
Z3
p4
q5
6
7
8
9
»10
Ό11
Ψ12
Ω13
υ14
φ15
16
17
 
 
 
Ξ
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28
29
30
 31
!32
"33
#34
$35
%36
&37
'38
(39
)40
*41
+42
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

B0
C1
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
 
 
 

Y0
Z1
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
 
 
 

p0
q1
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
 
 
 

0
1
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
 
 
 

0
1
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
 
 
 
 
 
 
 
 

»0
Ό1
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
 
 
 
 
 
 
 
 

Ψ0
Ω1
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
 
 
 
 
 
 
 
 

υ0
φ1
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
 
 
 
 
 
 
 
 

0
1
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

serving_default_imageInputPlaceholder*0
_output_shapes
:?????????`ΐ*
dtype0*%
shape:?????????`ΐ
ώ
StatefulPartitionedCallStatefulPartitionedCallserving_default_imageInputconv_1/kernelconv_1/biasbatch_normalization/gammabatch_normalization/betabatch_normalization/moving_mean#batch_normalization/moving_varianceconv_2/kernelconv_2/biasbatch_normalization_1/gammabatch_normalization_1/beta!batch_normalization_1/moving_mean%batch_normalization_1/moving_varianceconv_3/kernelconv_3/biasbatch_normalization_2/gammabatch_normalization_2/beta!batch_normalization_2/moving_mean%batch_normalization_2/moving_varianceconv_4/kernelconv_4/biasbatch_normalization_3/gammabatch_normalization_3/beta!batch_normalization_3/moving_mean%batch_normalization_3/moving_varianceconv_5/kernelconv_5/biasbatch_normalization_4/gammabatch_normalization_4/beta!batch_normalization_4/moving_mean%batch_normalization_4/moving_varianceupsample_6/kernelupsample_6/biasconv_6/kernelconv_6/biasbatch_normalization_5/gammabatch_normalization_5/beta!batch_normalization_5/moving_mean%batch_normalization_5/moving_varianceupsample_7/kernelupsample_7/biasconv_7/kernelconv_7/biasbatch_normalization_6/gammabatch_normalization_6/beta!batch_normalization_6/moving_mean%batch_normalization_6/moving_varianceupsample_8/kernelupsample_8/biasconv_8/kernelconv_8/biasbatch_normalization_7/gammabatch_normalization_7/beta!batch_normalization_7/moving_mean%batch_normalization_7/moving_varianceupsample_9/kernelupsample_9/biasconv_9/kernelconv_9/biasbatch_normalization_8/gammabatch_normalization_8/beta!batch_normalization_8/moving_mean%batch_normalization_8/moving_variancefinal/kernel
final/bias*L
TinE
C2A*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????`ΐ*b
_read_only_resource_inputsD
B@	
 !"#$%&'()*+,-./0123456789:;<=>?@*2
config_proto" 

CPU

GPU2 *0J 8 */
f*R(
&__inference_signature_wrapper_57380334
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 

StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename!conv_1/kernel/Read/ReadVariableOpconv_1/bias/Read/ReadVariableOp-batch_normalization/gamma/Read/ReadVariableOp,batch_normalization/beta/Read/ReadVariableOp3batch_normalization/moving_mean/Read/ReadVariableOp7batch_normalization/moving_variance/Read/ReadVariableOp!conv_2/kernel/Read/ReadVariableOpconv_2/bias/Read/ReadVariableOp/batch_normalization_1/gamma/Read/ReadVariableOp.batch_normalization_1/beta/Read/ReadVariableOp5batch_normalization_1/moving_mean/Read/ReadVariableOp9batch_normalization_1/moving_variance/Read/ReadVariableOp!conv_3/kernel/Read/ReadVariableOpconv_3/bias/Read/ReadVariableOp/batch_normalization_2/gamma/Read/ReadVariableOp.batch_normalization_2/beta/Read/ReadVariableOp5batch_normalization_2/moving_mean/Read/ReadVariableOp9batch_normalization_2/moving_variance/Read/ReadVariableOp!conv_4/kernel/Read/ReadVariableOpconv_4/bias/Read/ReadVariableOp/batch_normalization_3/gamma/Read/ReadVariableOp.batch_normalization_3/beta/Read/ReadVariableOp5batch_normalization_3/moving_mean/Read/ReadVariableOp9batch_normalization_3/moving_variance/Read/ReadVariableOp!conv_5/kernel/Read/ReadVariableOpconv_5/bias/Read/ReadVariableOp/batch_normalization_4/gamma/Read/ReadVariableOp.batch_normalization_4/beta/Read/ReadVariableOp5batch_normalization_4/moving_mean/Read/ReadVariableOp9batch_normalization_4/moving_variance/Read/ReadVariableOp%upsample_6/kernel/Read/ReadVariableOp#upsample_6/bias/Read/ReadVariableOp!conv_6/kernel/Read/ReadVariableOpconv_6/bias/Read/ReadVariableOp/batch_normalization_5/gamma/Read/ReadVariableOp.batch_normalization_5/beta/Read/ReadVariableOp5batch_normalization_5/moving_mean/Read/ReadVariableOp9batch_normalization_5/moving_variance/Read/ReadVariableOp%upsample_7/kernel/Read/ReadVariableOp#upsample_7/bias/Read/ReadVariableOp!conv_7/kernel/Read/ReadVariableOpconv_7/bias/Read/ReadVariableOp/batch_normalization_6/gamma/Read/ReadVariableOp.batch_normalization_6/beta/Read/ReadVariableOp5batch_normalization_6/moving_mean/Read/ReadVariableOp9batch_normalization_6/moving_variance/Read/ReadVariableOp%upsample_8/kernel/Read/ReadVariableOp#upsample_8/bias/Read/ReadVariableOp!conv_8/kernel/Read/ReadVariableOpconv_8/bias/Read/ReadVariableOp/batch_normalization_7/gamma/Read/ReadVariableOp.batch_normalization_7/beta/Read/ReadVariableOp5batch_normalization_7/moving_mean/Read/ReadVariableOp9batch_normalization_7/moving_variance/Read/ReadVariableOp%upsample_9/kernel/Read/ReadVariableOp#upsample_9/bias/Read/ReadVariableOp!conv_9/kernel/Read/ReadVariableOpconv_9/bias/Read/ReadVariableOp/batch_normalization_8/gamma/Read/ReadVariableOp.batch_normalization_8/beta/Read/ReadVariableOp5batch_normalization_8/moving_mean/Read/ReadVariableOp9batch_normalization_8/moving_variance/Read/ReadVariableOp final/kernel/Read/ReadVariableOpfinal/bias/Read/ReadVariableOpConst*M
TinF
D2B*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 **
f%R#
!__inference__traced_save_57382921

StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv_1/kernelconv_1/biasbatch_normalization/gammabatch_normalization/betabatch_normalization/moving_mean#batch_normalization/moving_varianceconv_2/kernelconv_2/biasbatch_normalization_1/gammabatch_normalization_1/beta!batch_normalization_1/moving_mean%batch_normalization_1/moving_varianceconv_3/kernelconv_3/biasbatch_normalization_2/gammabatch_normalization_2/beta!batch_normalization_2/moving_mean%batch_normalization_2/moving_varianceconv_4/kernelconv_4/biasbatch_normalization_3/gammabatch_normalization_3/beta!batch_normalization_3/moving_mean%batch_normalization_3/moving_varianceconv_5/kernelconv_5/biasbatch_normalization_4/gammabatch_normalization_4/beta!batch_normalization_4/moving_mean%batch_normalization_4/moving_varianceupsample_6/kernelupsample_6/biasconv_6/kernelconv_6/biasbatch_normalization_5/gammabatch_normalization_5/beta!batch_normalization_5/moving_mean%batch_normalization_5/moving_varianceupsample_7/kernelupsample_7/biasconv_7/kernelconv_7/biasbatch_normalization_6/gammabatch_normalization_6/beta!batch_normalization_6/moving_mean%batch_normalization_6/moving_varianceupsample_8/kernelupsample_8/biasconv_8/kernelconv_8/biasbatch_normalization_7/gammabatch_normalization_7/beta!batch_normalization_7/moving_mean%batch_normalization_7/moving_varianceupsample_9/kernelupsample_9/biasconv_9/kernelconv_9/biasbatch_normalization_8/gammabatch_normalization_8/beta!batch_normalization_8/moving_mean%batch_normalization_8/moving_variancefinal/kernel
final/bias*L
TinE
C2A*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *-
f(R&
$__inference__traced_restore_57383123Ν"
ή

S__inference_batch_normalization_3_layer_call_and_return_conditional_losses_57378689

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityu
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype02
ReadVariableOp_1¨
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ο
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:?????????:::::*
epsilon%o:*
is_training( 2
FusedBatchNormV3q
IdentityIdentityFusedBatchNormV3:y:0*
T0*0
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:?????????:::::X T
0
_output_shapes
:?????????
 
_user_specified_nameinputs


S__inference_batch_normalization_6_layer_call_and_return_conditional_losses_57382311

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ά
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%o:*
is_training( 2
FusedBatchNormV3
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????@:::::i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
Ή
F
*__inference_re_lu_6_layer_call_fn_57382347

inputs
identityΠ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????0@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_re_lu_6_layer_call_and_return_conditional_losses_573791092
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????0@2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????0@:W S
/
_output_shapes
:?????????0@
 
_user_specified_nameinputs
Ή
F
*__inference_re_lu_1_layer_call_fn_57381536

inputs
identityΠ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????0` * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_re_lu_1_layer_call_and_return_conditional_losses_573785042
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????0` 2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????0` :W S
/
_output_shapes
:?????????0` 
 
_user_specified_nameinputs
Λ
°
S__inference_batch_normalization_6_layer_call_and_return_conditional_losses_57382293

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity’AssignNewValue’AssignNewValue_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1κ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%o:*
exponential_avg_factor%
Χ#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1¦
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????@::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs


Q__inference_batch_normalization_layer_call_and_return_conditional_losses_57381279

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ά
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o:*
is_training( 2
FusedBatchNormV3
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????:::::i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
Χ
°
S__inference_batch_normalization_5_layer_call_and_return_conditional_losses_57382123

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity’AssignNewValue’AssignNewValue_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype02
ReadVariableOp_1¨
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ο
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,???????????????????????????:::::*
epsilon%o:*
exponential_avg_factor%
Χ#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1§
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*B
_output_shapes0
.:,???????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,???????????????????????????::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:j f
B
_output_shapes0
.:,???????????????????????????
 
_user_specified_nameinputs
Σ
a
E__inference_re_lu_6_layer_call_and_return_conditional_losses_57379109

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:?????????0@2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:?????????0@2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????0@:W S
/
_output_shapes
:?????????0@
 
_user_specified_nameinputs

°
S__inference_batch_normalization_8_layer_call_and_return_conditional_losses_57379316

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity’AssignNewValue’AssignNewValue_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ω
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*L
_output_shapes:
8:?????????`ΐ:::::*
epsilon%o:*
exponential_avg_factor%
Χ#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*0
_output_shapes
:?????????`ΐ2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:?????????`ΐ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:X T
0
_output_shapes
:?????????`ΐ
 
_user_specified_nameinputs
β
«
8__inference_batch_normalization_8_layer_call_fn_57382664

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity’StatefulPartitionedCall©
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????`ΐ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *\
fWRU
S__inference_batch_normalization_8_layer_call_and_return_conditional_losses_573793162
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:?????????`ΐ2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:?????????`ΐ::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:?????????`ΐ
 
_user_specified_nameinputs
Α
~
R__inference_tf_op_layer_concat_7_layer_call_and_return_conditional_losses_57382184
inputs_0
inputs_1
identityi
concat_7/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
concat_7/axis
concat_7ConcatV2inputs_0inputs_1concat_7/axis:output:0*
N*
T0*
_cloned(*0
_output_shapes
:?????????02

concat_7n
IdentityIdentityconcat_7:output:0*
T0*0
_output_shapes
:?????????02

Identity"
identityIdentity:output:0*[
_input_shapesJ
H:+???????????????????????????@:?????????0@:k g
A
_output_shapes/
-:+???????????????????????????@
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:?????????0@
"
_user_specified_name
inputs/1
ή

S__inference_batch_normalization_4_layer_call_and_return_conditional_losses_57378802

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityu
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype02
ReadVariableOp_1¨
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ο
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:?????????:::::*
epsilon%o:*
is_training( 2
FusedBatchNormV3q
IdentityIdentityFusedBatchNormV3:y:0*
T0*0
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:?????????:::::X T
0
_output_shapes
:?????????
 
_user_specified_nameinputs
Κ
κ
J__inference_functional_1_layer_call_and_return_conditional_losses_57379410

imageinput
conv_1_57378308
conv_1_57378310 
batch_normalization_57378377 
batch_normalization_57378379 
batch_normalization_57378381 
batch_normalization_57378383
conv_2_57378421
conv_2_57378423"
batch_normalization_1_57378490"
batch_normalization_1_57378492"
batch_normalization_1_57378494"
batch_normalization_1_57378496
conv_3_57378534
conv_3_57378536"
batch_normalization_2_57378603"
batch_normalization_2_57378605"
batch_normalization_2_57378607"
batch_normalization_2_57378609
conv_4_57378647
conv_4_57378649"
batch_normalization_3_57378716"
batch_normalization_3_57378718"
batch_normalization_3_57378720"
batch_normalization_3_57378722
conv_5_57378760
conv_5_57378762"
batch_normalization_4_57378829"
batch_normalization_4_57378831"
batch_normalization_4_57378833"
batch_normalization_4_57378835
upsample_6_57378851
upsample_6_57378853
conv_6_57378893
conv_6_57378895"
batch_normalization_5_57378962"
batch_normalization_5_57378964"
batch_normalization_5_57378966"
batch_normalization_5_57378968
upsample_7_57378984
upsample_7_57378986
conv_7_57379026
conv_7_57379028"
batch_normalization_6_57379095"
batch_normalization_6_57379097"
batch_normalization_6_57379099"
batch_normalization_6_57379101
upsample_8_57379117
upsample_8_57379119
conv_8_57379159
conv_8_57379161"
batch_normalization_7_57379228"
batch_normalization_7_57379230"
batch_normalization_7_57379232"
batch_normalization_7_57379234
upsample_9_57379250
upsample_9_57379252
conv_9_57379292
conv_9_57379294"
batch_normalization_8_57379361"
batch_normalization_8_57379363"
batch_normalization_8_57379365"
batch_normalization_8_57379367
final_57379404
final_57379406
identity’+batch_normalization/StatefulPartitionedCall’-batch_normalization_1/StatefulPartitionedCall’-batch_normalization_2/StatefulPartitionedCall’-batch_normalization_3/StatefulPartitionedCall’-batch_normalization_4/StatefulPartitionedCall’-batch_normalization_5/StatefulPartitionedCall’-batch_normalization_6/StatefulPartitionedCall’-batch_normalization_7/StatefulPartitionedCall’-batch_normalization_8/StatefulPartitionedCall’conv_1/StatefulPartitionedCall’conv_2/StatefulPartitionedCall’conv_3/StatefulPartitionedCall’conv_4/StatefulPartitionedCall’conv_5/StatefulPartitionedCall’conv_6/StatefulPartitionedCall’conv_7/StatefulPartitionedCall’conv_8/StatefulPartitionedCall’conv_9/StatefulPartitionedCall’final/StatefulPartitionedCall’"upsample_6/StatefulPartitionedCall’"upsample_7/StatefulPartitionedCall’"upsample_8/StatefulPartitionedCall’"upsample_9/StatefulPartitionedCall
#tf_op_layer_RealDiv/PartitionedCallPartitionedCall
imageinput*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????`ΐ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *Z
fURS
Q__inference_tf_op_layer_RealDiv_layer_call_and_return_conditional_losses_573782652%
#tf_op_layer_RealDiv/PartitionedCall
tf_op_layer_Sub/PartitionedCallPartitionedCall,tf_op_layer_RealDiv/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????`ΐ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *V
fQRO
M__inference_tf_op_layer_Sub_layer_call_and_return_conditional_losses_573782792!
tf_op_layer_Sub/PartitionedCallΐ
conv_1/StatefulPartitionedCallStatefulPartitionedCall(tf_op_layer_Sub/PartitionedCall:output:0conv_1_57378308conv_1_57378310*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????`ΐ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *M
fHRF
D__inference_conv_1_layer_call_and_return_conditional_losses_573782972 
conv_1/StatefulPartitionedCallΎ
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall'conv_1/StatefulPartitionedCall:output:0batch_normalization_57378377batch_normalization_57378379batch_normalization_57378381batch_normalization_57378383*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????`ΐ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *Z
fURS
Q__inference_batch_normalization_layer_call_and_return_conditional_losses_573783322-
+batch_normalization/StatefulPartitionedCall
re_lu/PartitionedCallPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????`ΐ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *L
fGRE
C__inference_re_lu_layer_call_and_return_conditional_losses_573783912
re_lu/PartitionedCall
max_pooling2d/PartitionedCallPartitionedCallre_lu/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????0`* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *T
fORM
K__inference_max_pooling2d_layer_call_and_return_conditional_losses_573771892
max_pooling2d/PartitionedCall½
conv_2/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0conv_2_57378421conv_2_57378423*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????0` *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *M
fHRF
D__inference_conv_2_layer_call_and_return_conditional_losses_573784102 
conv_2/StatefulPartitionedCallΛ
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall'conv_2/StatefulPartitionedCall:output:0batch_normalization_1_57378490batch_normalization_1_57378492batch_normalization_1_57378494batch_normalization_1_57378496*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????0` *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *\
fWRU
S__inference_batch_normalization_1_layer_call_and_return_conditional_losses_573784452/
-batch_normalization_1/StatefulPartitionedCall
re_lu_1/PartitionedCallPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????0` * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_re_lu_1_layer_call_and_return_conditional_losses_573785042
re_lu_1/PartitionedCall
max_pooling2d_1/PartitionedCallPartitionedCall re_lu_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????0 * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *V
fQRO
M__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_573773052!
max_pooling2d_1/PartitionedCallΏ
conv_3/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_1/PartitionedCall:output:0conv_3_57378534conv_3_57378536*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????0@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *M
fHRF
D__inference_conv_3_layer_call_and_return_conditional_losses_573785232 
conv_3/StatefulPartitionedCallΛ
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall'conv_3/StatefulPartitionedCall:output:0batch_normalization_2_57378603batch_normalization_2_57378605batch_normalization_2_57378607batch_normalization_2_57378609*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????0@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *\
fWRU
S__inference_batch_normalization_2_layer_call_and_return_conditional_losses_573785582/
-batch_normalization_2/StatefulPartitionedCall
re_lu_2/PartitionedCallPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????0@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_re_lu_2_layer_call_and_return_conditional_losses_573786172
re_lu_2/PartitionedCall
max_pooling2d_2/PartitionedCallPartitionedCall re_lu_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *V
fQRO
M__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_573774212!
max_pooling2d_2/PartitionedCallΐ
conv_4/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_2/PartitionedCall:output:0conv_4_57378647conv_4_57378649*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *M
fHRF
D__inference_conv_4_layer_call_and_return_conditional_losses_573786362 
conv_4/StatefulPartitionedCallΜ
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCall'conv_4/StatefulPartitionedCall:output:0batch_normalization_3_57378716batch_normalization_3_57378718batch_normalization_3_57378720batch_normalization_3_57378722*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *\
fWRU
S__inference_batch_normalization_3_layer_call_and_return_conditional_losses_573786712/
-batch_normalization_3/StatefulPartitionedCall
re_lu_3/PartitionedCallPartitionedCall6batch_normalization_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_re_lu_3_layer_call_and_return_conditional_losses_573787302
re_lu_3/PartitionedCall
max_pooling2d_3/PartitionedCallPartitionedCall re_lu_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *V
fQRO
M__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_573775372!
max_pooling2d_3/PartitionedCallΐ
conv_5/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_3/PartitionedCall:output:0conv_5_57378760conv_5_57378762*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *M
fHRF
D__inference_conv_5_layer_call_and_return_conditional_losses_573787492 
conv_5/StatefulPartitionedCallΜ
-batch_normalization_4/StatefulPartitionedCallStatefulPartitionedCall'conv_5/StatefulPartitionedCall:output:0batch_normalization_4_57378829batch_normalization_4_57378831batch_normalization_4_57378833batch_normalization_4_57378835*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *\
fWRU
S__inference_batch_normalization_4_layer_call_and_return_conditional_losses_573787842/
-batch_normalization_4/StatefulPartitionedCall
re_lu_4/PartitionedCallPartitionedCall6batch_normalization_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_re_lu_4_layer_call_and_return_conditional_losses_573788432
re_lu_4/PartitionedCallή
"upsample_6/StatefulPartitionedCallStatefulPartitionedCall re_lu_4/PartitionedCall:output:0upsample_6_57378851upsample_6_57378853*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,???????????????????????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *Q
fLRJ
H__inference_upsample_6_layer_call_and_return_conditional_losses_573776852$
"upsample_6/StatefulPartitionedCallΠ
$tf_op_layer_concat_6/PartitionedCallPartitionedCall+upsample_6/StatefulPartitionedCall:output:0 re_lu_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *[
fVRT
R__inference_tf_op_layer_concat_6_layer_call_and_return_conditional_losses_573788632&
$tf_op_layer_concat_6/PartitionedCallΕ
conv_6/StatefulPartitionedCallStatefulPartitionedCall-tf_op_layer_concat_6/PartitionedCall:output:0conv_6_57378893conv_6_57378895*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *M
fHRF
D__inference_conv_6_layer_call_and_return_conditional_losses_573788822 
conv_6/StatefulPartitionedCallΜ
-batch_normalization_5/StatefulPartitionedCallStatefulPartitionedCall'conv_6/StatefulPartitionedCall:output:0batch_normalization_5_57378962batch_normalization_5_57378964batch_normalization_5_57378966batch_normalization_5_57378968*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *\
fWRU
S__inference_batch_normalization_5_layer_call_and_return_conditional_losses_573789172/
-batch_normalization_5/StatefulPartitionedCall
re_lu_5/PartitionedCallPartitionedCall6batch_normalization_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_re_lu_5_layer_call_and_return_conditional_losses_573789762
re_lu_5/PartitionedCallέ
"upsample_7/StatefulPartitionedCallStatefulPartitionedCall re_lu_5/PartitionedCall:output:0upsample_7_57378984upsample_7_57378986*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *Q
fLRJ
H__inference_upsample_7_layer_call_and_return_conditional_losses_573778372$
"upsample_7/StatefulPartitionedCallΠ
$tf_op_layer_concat_7/PartitionedCallPartitionedCall+upsample_7/StatefulPartitionedCall:output:0 re_lu_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????0* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *[
fVRT
R__inference_tf_op_layer_concat_7_layer_call_and_return_conditional_losses_573789962&
$tf_op_layer_concat_7/PartitionedCallΔ
conv_7/StatefulPartitionedCallStatefulPartitionedCall-tf_op_layer_concat_7/PartitionedCall:output:0conv_7_57379026conv_7_57379028*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????0@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *M
fHRF
D__inference_conv_7_layer_call_and_return_conditional_losses_573790152 
conv_7/StatefulPartitionedCallΛ
-batch_normalization_6/StatefulPartitionedCallStatefulPartitionedCall'conv_7/StatefulPartitionedCall:output:0batch_normalization_6_57379095batch_normalization_6_57379097batch_normalization_6_57379099batch_normalization_6_57379101*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????0@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *\
fWRU
S__inference_batch_normalization_6_layer_call_and_return_conditional_losses_573790502/
-batch_normalization_6/StatefulPartitionedCall
re_lu_6/PartitionedCallPartitionedCall6batch_normalization_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????0@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_re_lu_6_layer_call_and_return_conditional_losses_573791092
re_lu_6/PartitionedCallέ
"upsample_8/StatefulPartitionedCallStatefulPartitionedCall re_lu_6/PartitionedCall:output:0upsample_8_57379117upsample_8_57379119*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *Q
fLRJ
H__inference_upsample_8_layer_call_and_return_conditional_losses_573779892$
"upsample_8/StatefulPartitionedCallΟ
$tf_op_layer_concat_8/PartitionedCallPartitionedCall+upsample_8/StatefulPartitionedCall:output:0 re_lu_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????0`@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *[
fVRT
R__inference_tf_op_layer_concat_8_layer_call_and_return_conditional_losses_573791292&
$tf_op_layer_concat_8/PartitionedCallΔ
conv_8/StatefulPartitionedCallStatefulPartitionedCall-tf_op_layer_concat_8/PartitionedCall:output:0conv_8_57379159conv_8_57379161*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????0` *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *M
fHRF
D__inference_conv_8_layer_call_and_return_conditional_losses_573791482 
conv_8/StatefulPartitionedCallΛ
-batch_normalization_7/StatefulPartitionedCallStatefulPartitionedCall'conv_8/StatefulPartitionedCall:output:0batch_normalization_7_57379228batch_normalization_7_57379230batch_normalization_7_57379232batch_normalization_7_57379234*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????0` *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *\
fWRU
S__inference_batch_normalization_7_layer_call_and_return_conditional_losses_573791832/
-batch_normalization_7/StatefulPartitionedCall
re_lu_7/PartitionedCallPartitionedCall6batch_normalization_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????0` * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_re_lu_7_layer_call_and_return_conditional_losses_573792422
re_lu_7/PartitionedCallέ
"upsample_9/StatefulPartitionedCallStatefulPartitionedCall re_lu_7/PartitionedCall:output:0upsample_9_57379250upsample_9_57379252*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *Q
fLRJ
H__inference_upsample_9_layer_call_and_return_conditional_losses_573781412$
"upsample_9/StatefulPartitionedCallΞ
$tf_op_layer_concat_9/PartitionedCallPartitionedCall+upsample_9/StatefulPartitionedCall:output:0re_lu/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????`ΐ * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *[
fVRT
R__inference_tf_op_layer_concat_9_layer_call_and_return_conditional_losses_573792622&
$tf_op_layer_concat_9/PartitionedCallΕ
conv_9/StatefulPartitionedCallStatefulPartitionedCall-tf_op_layer_concat_9/PartitionedCall:output:0conv_9_57379292conv_9_57379294*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????`ΐ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *M
fHRF
D__inference_conv_9_layer_call_and_return_conditional_losses_573792812 
conv_9/StatefulPartitionedCallΜ
-batch_normalization_8/StatefulPartitionedCallStatefulPartitionedCall'conv_9/StatefulPartitionedCall:output:0batch_normalization_8_57379361batch_normalization_8_57379363batch_normalization_8_57379365batch_normalization_8_57379367*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????`ΐ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *\
fWRU
S__inference_batch_normalization_8_layer_call_and_return_conditional_losses_573793162/
-batch_normalization_8/StatefulPartitionedCall
re_lu_8/PartitionedCallPartitionedCall6batch_normalization_8/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????`ΐ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_re_lu_8_layer_call_and_return_conditional_losses_573793752
re_lu_8/PartitionedCall³
final/StatefulPartitionedCallStatefulPartitionedCall re_lu_8/PartitionedCall:output:0final_57379404final_57379406*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????`ΐ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *L
fGRE
C__inference_final_layer_call_and_return_conditional_losses_573793932
final/StatefulPartitionedCall
IdentityIdentity&final/StatefulPartitionedCall:output:0,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall.^batch_normalization_3/StatefulPartitionedCall.^batch_normalization_4/StatefulPartitionedCall.^batch_normalization_5/StatefulPartitionedCall.^batch_normalization_6/StatefulPartitionedCall.^batch_normalization_7/StatefulPartitionedCall.^batch_normalization_8/StatefulPartitionedCall^conv_1/StatefulPartitionedCall^conv_2/StatefulPartitionedCall^conv_3/StatefulPartitionedCall^conv_4/StatefulPartitionedCall^conv_5/StatefulPartitionedCall^conv_6/StatefulPartitionedCall^conv_7/StatefulPartitionedCall^conv_8/StatefulPartitionedCall^conv_9/StatefulPartitionedCall^final/StatefulPartitionedCall#^upsample_6/StatefulPartitionedCall#^upsample_7/StatefulPartitionedCall#^upsample_8/StatefulPartitionedCall#^upsample_9/StatefulPartitionedCall*
T0*0
_output_shapes
:?????????`ΐ2

Identity"
identityIdentity:output:0*±
_input_shapes
:?????????`ΐ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2^
-batch_normalization_3/StatefulPartitionedCall-batch_normalization_3/StatefulPartitionedCall2^
-batch_normalization_4/StatefulPartitionedCall-batch_normalization_4/StatefulPartitionedCall2^
-batch_normalization_5/StatefulPartitionedCall-batch_normalization_5/StatefulPartitionedCall2^
-batch_normalization_6/StatefulPartitionedCall-batch_normalization_6/StatefulPartitionedCall2^
-batch_normalization_7/StatefulPartitionedCall-batch_normalization_7/StatefulPartitionedCall2^
-batch_normalization_8/StatefulPartitionedCall-batch_normalization_8/StatefulPartitionedCall2@
conv_1/StatefulPartitionedCallconv_1/StatefulPartitionedCall2@
conv_2/StatefulPartitionedCallconv_2/StatefulPartitionedCall2@
conv_3/StatefulPartitionedCallconv_3/StatefulPartitionedCall2@
conv_4/StatefulPartitionedCallconv_4/StatefulPartitionedCall2@
conv_5/StatefulPartitionedCallconv_5/StatefulPartitionedCall2@
conv_6/StatefulPartitionedCallconv_6/StatefulPartitionedCall2@
conv_7/StatefulPartitionedCallconv_7/StatefulPartitionedCall2@
conv_8/StatefulPartitionedCallconv_8/StatefulPartitionedCall2@
conv_9/StatefulPartitionedCallconv_9/StatefulPartitionedCall2>
final/StatefulPartitionedCallfinal/StatefulPartitionedCall2H
"upsample_6/StatefulPartitionedCall"upsample_6/StatefulPartitionedCall2H
"upsample_7/StatefulPartitionedCall"upsample_7/StatefulPartitionedCall2H
"upsample_8/StatefulPartitionedCall"upsample_8/StatefulPartitionedCall2H
"upsample_9/StatefulPartitionedCall"upsample_9/StatefulPartitionedCall:\ X
0
_output_shapes
:?????????`ΐ
$
_user_specified_name
imageInput

°
S__inference_batch_normalization_6_layer_call_and_return_conditional_losses_57382229

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity’AssignNewValue’AssignNewValue_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ψ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????0@:@:@:@:@:*
epsilon%o:*
exponential_avg_factor%
Χ#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*/
_output_shapes
:?????????0@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????0@::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:W S
/
_output_shapes
:?????????0@
 
_user_specified_nameinputs
ή

S__inference_batch_normalization_4_layer_call_and_return_conditional_losses_57381907

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityu
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype02
ReadVariableOp_1¨
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ο
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:?????????:::::*
epsilon%o:*
is_training( 2
FusedBatchNormV3q
IdentityIdentityFusedBatchNormV3:y:0*
T0*0
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:?????????:::::X T
0
_output_shapes
:?????????
 
_user_specified_nameinputs
ή

S__inference_batch_normalization_3_layer_call_and_return_conditional_losses_57381814

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityu
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype02
ReadVariableOp_1¨
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ο
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:?????????:::::*
epsilon%o:*
is_training( 2
FusedBatchNormV3q
IdentityIdentityFusedBatchNormV3:y:0*
T0*0
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:?????????:::::X T
0
_output_shapes
:?????????
 
_user_specified_nameinputs
¨
«
8__inference_batch_normalization_8_layer_call_fn_57382613

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity’StatefulPartitionedCallΌ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *\
fWRU
S__inference_batch_normalization_8_layer_call_and_return_conditional_losses_573782442
StatefulPartitionedCall¨
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
ΰ
«
8__inference_batch_normalization_2_layer_call_fn_57381619

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity’StatefulPartitionedCallͺ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????0@*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *\
fWRU
S__inference_batch_normalization_2_layer_call_and_return_conditional_losses_573785762
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????0@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????0@::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????0@
 
_user_specified_nameinputs
¨
«
8__inference_batch_normalization_6_layer_call_fn_57382337

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity’StatefulPartitionedCallΌ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *\
fWRU
S__inference_batch_normalization_6_layer_call_and_return_conditional_losses_573779402
StatefulPartitionedCall¨
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????@::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs


S__inference_batch_normalization_2_layer_call_and_return_conditional_losses_57381657

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ά
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%o:*
is_training( 2
FusedBatchNormV3
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????@:::::i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs

°
S__inference_batch_normalization_6_layer_call_and_return_conditional_losses_57379050

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity’AssignNewValue’AssignNewValue_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ψ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????0@:@:@:@:@:*
epsilon%o:*
exponential_avg_factor%
Χ#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*/
_output_shapes
:?????????0@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????0@::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:W S
/
_output_shapes
:?????????0@
 
_user_specified_nameinputs
₯
¬
D__inference_conv_3_layer_call_and_return_conditional_losses_57381546

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02
Conv2D/ReadVariableOp£
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????0@*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????0@2	
BiasAddl
IdentityIdentityBiasAdd:output:0*
T0*/
_output_shapes
:?????????0@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????0 :::W S
/
_output_shapes
:?????????0 
 
_user_specified_nameinputs
δ
«
8__inference_batch_normalization_5_layer_call_fn_57382103

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity’StatefulPartitionedCall«
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *\
fWRU
S__inference_batch_normalization_5_layer_call_and_return_conditional_losses_573789352
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:?????????::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:?????????
 
_user_specified_nameinputs

?
Q__inference_batch_normalization_layer_call_and_return_conditional_losses_57381325

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity’AssignNewValue’AssignNewValue_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ω
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*L
_output_shapes:
8:?????????`ΐ:::::*
epsilon%o:*
exponential_avg_factor%
Χ#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*0
_output_shapes
:?????????`ΐ2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:?????????`ΐ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:X T
0
_output_shapes
:?????????`ΐ
 
_user_specified_nameinputs


S__inference_batch_normalization_7_layer_call_and_return_conditional_losses_57378092

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ά
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+??????????????????????????? : : : : :*
epsilon%o:*
is_training( 2
FusedBatchNormV3
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+??????????????????????????? :::::i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs

~
)__inference_conv_7_layer_call_fn_57382209

inputs
unknown
	unknown_0
identity’StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????0@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *M
fHRF
D__inference_conv_7_layer_call_and_return_conditional_losses_573790152
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????0@2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:?????????0::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:?????????0
 
_user_specified_nameinputs
©
«
C__inference_final_layer_call_and_return_conditional_losses_57379393

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp€
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????`ΐ*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????`ΐ2	
BiasAddm
IdentityIdentityBiasAdd:output:0*
T0*0
_output_shapes
:?????????`ΐ2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:?????????`ΐ:::X T
0
_output_shapes
:?????????`ΐ
 
_user_specified_nameinputs

~
)__inference_conv_5_layer_call_fn_57381869

inputs
unknown
	unknown_0
identity’StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *M
fHRF
D__inference_conv_5_layer_call_and_return_conditional_losses_573787492
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:?????????::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:?????????
 
_user_specified_nameinputs
ΰ
«
8__inference_batch_normalization_1_layer_call_fn_57381526

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity’StatefulPartitionedCallͺ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????0` *&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *\
fWRU
S__inference_batch_normalization_1_layer_call_and_return_conditional_losses_573784632
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????0` 2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????0` ::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????0` 
 
_user_specified_nameinputs


S__inference_batch_normalization_6_layer_call_and_return_conditional_losses_57377940

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ά
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%o:*
is_training( 2
FusedBatchNormV3
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????@:::::i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
Ή
F
*__inference_re_lu_2_layer_call_fn_57381693

inputs
identityΠ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????0@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_re_lu_2_layer_call_and_return_conditional_losses_573786172
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????0@2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????0@:W S
/
_output_shapes
:?????????0@
 
_user_specified_nameinputs
Χ
a
E__inference_re_lu_8_layer_call_and_return_conditional_losses_57382682

inputs
identityW
ReluReluinputs*
T0*0
_output_shapes
:?????????`ΐ2
Reluo
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:?????????`ΐ2

Identity"
identityIdentity:output:0*/
_input_shapes
:?????????`ΐ:X T
0
_output_shapes
:?????????`ΐ
 
_user_specified_nameinputs
Χ
a
E__inference_re_lu_5_layer_call_and_return_conditional_losses_57382172

inputs
identityW
ReluReluinputs*
T0*0
_output_shapes
:?????????2
Reluo
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:?????????:X T
0
_output_shapes
:?????????
 
_user_specified_nameinputs
¨
¬
D__inference_conv_7_layer_call_and_return_conditional_losses_57379015

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@*
dtype02
Conv2D/ReadVariableOp£
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????0@*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????0@2	
BiasAddl
IdentityIdentityBiasAdd:output:0*
T0*/
_output_shapes
:?????????0@2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:?????????0:::X T
0
_output_shapes
:?????????0
 
_user_specified_nameinputs
¦
«
8__inference_batch_normalization_1_layer_call_fn_57381449

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity’StatefulPartitionedCallΊ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *\
fWRU
S__inference_batch_normalization_1_layer_call_and_return_conditional_losses_573772572
StatefulPartitionedCall¨
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+??????????????????????????? ::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?

S__inference_batch_normalization_2_layer_call_and_return_conditional_losses_57378576

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Κ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????0@:@:@:@:@:*
epsilon%o:*
is_training( 2
FusedBatchNormV3p
IdentityIdentityFusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????0@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????0@:::::W S
/
_output_shapes
:?????????0@
 
_user_specified_nameinputs
¬
«
8__inference_batch_normalization_5_layer_call_fn_57382167

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity’StatefulPartitionedCall½
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,???????????????????????????*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *\
fWRU
S__inference_batch_normalization_5_layer_call_and_return_conditional_losses_573777882
StatefulPartitionedCall©
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,???????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,???????????????????????????::::22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,???????????????????????????
 
_user_specified_nameinputs
’
©
6__inference_batch_normalization_layer_call_fn_57381292

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity’StatefulPartitionedCallΈ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *Z
fURS
Q__inference_batch_normalization_layer_call_and_return_conditional_losses_573771412
StatefulPartitionedCall¨
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
­
¬
D__inference_conv_5_layer_call_and_return_conditional_losses_57378749

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp€
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????2	
BiasAddm
IdentityIdentityBiasAdd:output:0*
T0*0
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:?????????:::X T
0
_output_shapes
:?????????
 
_user_specified_nameinputs
¦
«
8__inference_batch_normalization_8_layer_call_fn_57382600

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity’StatefulPartitionedCallΊ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *\
fWRU
S__inference_batch_normalization_8_layer_call_and_return_conditional_losses_573782132
StatefulPartitionedCall¨
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
Χ
a
E__inference_re_lu_5_layer_call_and_return_conditional_losses_57378976

inputs
identityW
ReluReluinputs*
T0*0
_output_shapes
:?????????2
Reluo
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:?????????:X T
0
_output_shapes
:?????????
 
_user_specified_nameinputs
¬
«
8__inference_batch_normalization_4_layer_call_fn_57381997

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity’StatefulPartitionedCall½
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,???????????????????????????*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *\
fWRU
S__inference_batch_normalization_4_layer_call_and_return_conditional_losses_573776362
StatefulPartitionedCall©
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,???????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,???????????????????????????::::22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,???????????????????????????
 
_user_specified_nameinputs
Χ
°
S__inference_batch_normalization_5_layer_call_and_return_conditional_losses_57377757

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity’AssignNewValue’AssignNewValue_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype02
ReadVariableOp_1¨
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ο
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,???????????????????????????:::::*
epsilon%o:*
exponential_avg_factor%
Χ#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1§
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*B
_output_shapes0
.:,???????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,???????????????????????????::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:j f
B
_output_shapes0
.:,???????????????????????????
 
_user_specified_nameinputs

°
S__inference_batch_normalization_5_layer_call_and_return_conditional_losses_57382059

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity’AssignNewValue’AssignNewValue_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype02
ReadVariableOp_1¨
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1έ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:?????????:::::*
epsilon%o:*
exponential_avg_factor%
Χ#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*0
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:?????????::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:X T
0
_output_shapes
:?????????
 
_user_specified_nameinputs
Ώ
Ϋ
/__inference_functional_1_layer_call_fn_57381200

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

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28

unknown_29

unknown_30

unknown_31

unknown_32

unknown_33

unknown_34

unknown_35

unknown_36

unknown_37

unknown_38

unknown_39

unknown_40

unknown_41

unknown_42

unknown_43

unknown_44

unknown_45

unknown_46

unknown_47

unknown_48

unknown_49

unknown_50

unknown_51

unknown_52

unknown_53

unknown_54

unknown_55

unknown_56

unknown_57

unknown_58

unknown_59

unknown_60

unknown_61

unknown_62
identity’StatefulPartitionedCallγ	
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46
unknown_47
unknown_48
unknown_49
unknown_50
unknown_51
unknown_52
unknown_53
unknown_54
unknown_55
unknown_56
unknown_57
unknown_58
unknown_59
unknown_60
unknown_61
unknown_62*L
TinE
C2A*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????`ΐ*b
_read_only_resource_inputsD
B@	
 !"#$%&'()*+,-./0123456789:;<=>?@*2
config_proto" 

CPU

GPU2 *0J 8 *S
fNRL
J__inference_functional_1_layer_call_and_return_conditional_losses_573800682
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:?????????`ΐ2

Identity"
identityIdentity:output:0*±
_input_shapes
:?????????`ΐ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:?????????`ΐ
 
_user_specified_nameinputs
δ
«
8__inference_batch_normalization_3_layer_call_fn_57381840

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity’StatefulPartitionedCall«
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *\
fWRU
S__inference_batch_normalization_3_layer_call_and_return_conditional_losses_573786892
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:?????????::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:?????????
 
_user_specified_nameinputs

°
S__inference_batch_normalization_4_layer_call_and_return_conditional_losses_57381889

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity’AssignNewValue’AssignNewValue_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype02
ReadVariableOp_1¨
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1έ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:?????????:::::*
epsilon%o:*
exponential_avg_factor%
Χ#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*0
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:?????????::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:X T
0
_output_shapes
:?????????
 
_user_specified_nameinputs

i
M__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_57377421

inputs
identity­
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
2	
MaxPool
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
₯
¬
D__inference_conv_8_layer_call_and_return_conditional_losses_57382370

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@ *
dtype02
Conv2D/ReadVariableOp£
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????0` *
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????0` 2	
BiasAddl
IdentityIdentityBiasAdd:output:0*
T0*/
_output_shapes
:?????????0` 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????0`@:::W S
/
_output_shapes
:?????????0`@
 
_user_specified_nameinputs

°
S__inference_batch_normalization_8_layer_call_and_return_conditional_losses_57382633

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity’AssignNewValue’AssignNewValue_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ω
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*L
_output_shapes:
8:?????????`ΐ:::::*
epsilon%o:*
exponential_avg_factor%
Χ#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*0
_output_shapes
:?????????`ΐ2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:?????????`ΐ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:X T
0
_output_shapes
:?????????`ΐ
 
_user_specified_nameinputs
Φ

S__inference_batch_normalization_8_layer_call_and_return_conditional_losses_57382651

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Λ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*L
_output_shapes:
8:?????????`ΐ:::::*
epsilon%o:*
is_training( 2
FusedBatchNormV3q
IdentityIdentityFusedBatchNormV3:y:0*
T0*0
_output_shapes
:?????????`ΐ2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:?????????`ΐ:::::X T
0
_output_shapes
:?????????`ΐ
 
_user_specified_nameinputs
ͺ
¬
D__inference_conv_1_layer_call_and_return_conditional_losses_57381232

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp€
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????`ΐ*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????`ΐ2	
BiasAddm
IdentityIdentityBiasAdd:output:0*
T0*0
_output_shapes
:?????????`ΐ2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:?????????`ΐ:::X T
0
_output_shapes
:?????????`ΐ
 
_user_specified_nameinputs
Ε
~
R__inference_tf_op_layer_concat_6_layer_call_and_return_conditional_losses_57382014
inputs_0
inputs_1
identityi
concat_6/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
concat_6/axis
concat_6ConcatV2inputs_0inputs_1concat_6/axis:output:0*
N*
T0*
_cloned(*0
_output_shapes
:?????????2

concat_6n
IdentityIdentityconcat_6:output:0*
T0*0
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*]
_input_shapesL
J:,???????????????????????????:?????????:l h
B
_output_shapes0
.:,???????????????????????????
"
_user_specified_name
inputs/0:ZV
0
_output_shapes
:?????????
"
_user_specified_name
inputs/1
Γ
~
R__inference_tf_op_layer_concat_9_layer_call_and_return_conditional_losses_57382524
inputs_0
inputs_1
identityi
concat_9/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
concat_9/axis
concat_9ConcatV2inputs_0inputs_1concat_9/axis:output:0*
N*
T0*
_cloned(*0
_output_shapes
:?????????`ΐ 2

concat_9n
IdentityIdentityconcat_9:output:0*
T0*0
_output_shapes
:?????????`ΐ 2

Identity"
identityIdentity:output:0*\
_input_shapesK
I:+???????????????????????????:?????????`ΐ:k g
A
_output_shapes/
-:+???????????????????????????
"
_user_specified_name
inputs/0:ZV
0
_output_shapes
:?????????`ΐ
"
_user_specified_name
inputs/1
ΰ
«
8__inference_batch_normalization_7_layer_call_fn_57382507

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity’StatefulPartitionedCallͺ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????0` *&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *\
fWRU
S__inference_batch_normalization_7_layer_call_and_return_conditional_losses_573792012
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????0` 2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????0` ::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????0` 
 
_user_specified_nameinputs
·
|
R__inference_tf_op_layer_concat_8_layer_call_and_return_conditional_losses_57379129

inputs
inputs_1
identityi
concat_8/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
concat_8/axis
concat_8ConcatV2inputsinputs_1concat_8/axis:output:0*
N*
T0*
_cloned(*/
_output_shapes
:?????????0`@2

concat_8m
IdentityIdentityconcat_8:output:0*
T0*/
_output_shapes
:?????????0`@2

Identity"
identityIdentity:output:0*[
_input_shapesJ
H:+??????????????????????????? :?????????0` :i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs:WS
/
_output_shapes
:?????????0` 
 
_user_specified_nameinputs
Σ
a
E__inference_re_lu_1_layer_call_and_return_conditional_losses_57381531

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:?????????0` 2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:?????????0` 2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????0` :W S
/
_output_shapes
:?????????0` 
 
_user_specified_nameinputs
Ή
|
R__inference_tf_op_layer_concat_7_layer_call_and_return_conditional_losses_57378996

inputs
inputs_1
identityi
concat_7/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
concat_7/axis
concat_7ConcatV2inputsinputs_1concat_7/axis:output:0*
N*
T0*
_cloned(*0
_output_shapes
:?????????02

concat_7n
IdentityIdentityconcat_7:output:0*
T0*0
_output_shapes
:?????????02

Identity"
identityIdentity:output:0*[
_input_shapesJ
H:+???????????????????????????@:?????????0@:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs:WS
/
_output_shapes
:?????????0@
 
_user_specified_nameinputs
½
|
R__inference_tf_op_layer_concat_6_layer_call_and_return_conditional_losses_57378863

inputs
inputs_1
identityi
concat_6/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
concat_6/axis
concat_6ConcatV2inputsinputs_1concat_6/axis:output:0*
N*
T0*
_cloned(*0
_output_shapes
:?????????2

concat_6n
IdentityIdentityconcat_6:output:0*
T0*0
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*]
_input_shapesL
J:,???????????????????????????:?????????:j f
B
_output_shapes0
.:,???????????????????????????
 
_user_specified_nameinputs:XT
0
_output_shapes
:?????????
 
_user_specified_nameinputs
Λ
°
S__inference_batch_normalization_2_layer_call_and_return_conditional_losses_57377373

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity’AssignNewValue’AssignNewValue_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1κ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%o:*
exponential_avg_factor%
Χ#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1¦
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????@::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
Λ
°
S__inference_batch_normalization_7_layer_call_and_return_conditional_losses_57378061

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity’AssignNewValue’AssignNewValue_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1κ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+??????????????????????????? : : : : :*
epsilon%o:*
exponential_avg_factor%
Χ#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1¦
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+??????????????????????????? ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
©
«
C__inference_final_layer_call_and_return_conditional_losses_57382697

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp€
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????`ΐ*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????`ΐ2	
BiasAddm
IdentityIdentityBiasAdd:output:0*
T0*0
_output_shapes
:?????????`ΐ2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:?????????`ΐ:::X T
0
_output_shapes
:?????????`ΐ
 
_user_specified_nameinputs

°
S__inference_batch_normalization_7_layer_call_and_return_conditional_losses_57379183

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity’AssignNewValue’AssignNewValue_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ψ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????0` : : : : :*
epsilon%o:*
exponential_avg_factor%
Χ#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*/
_output_shapes
:?????????0` 2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????0` ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:W S
/
_output_shapes
:?????????0` 
 
_user_specified_nameinputs
υ
·!
J__inference_functional_1_layer_call_and_return_conditional_losses_57380643

inputs)
%conv_1_conv2d_readvariableop_resource*
&conv_1_biasadd_readvariableop_resource/
+batch_normalization_readvariableop_resource1
-batch_normalization_readvariableop_1_resource@
<batch_normalization_fusedbatchnormv3_readvariableop_resourceB
>batch_normalization_fusedbatchnormv3_readvariableop_1_resource)
%conv_2_conv2d_readvariableop_resource*
&conv_2_biasadd_readvariableop_resource1
-batch_normalization_1_readvariableop_resource3
/batch_normalization_1_readvariableop_1_resourceB
>batch_normalization_1_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource)
%conv_3_conv2d_readvariableop_resource*
&conv_3_biasadd_readvariableop_resource1
-batch_normalization_2_readvariableop_resource3
/batch_normalization_2_readvariableop_1_resourceB
>batch_normalization_2_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource)
%conv_4_conv2d_readvariableop_resource*
&conv_4_biasadd_readvariableop_resource1
-batch_normalization_3_readvariableop_resource3
/batch_normalization_3_readvariableop_1_resourceB
>batch_normalization_3_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource)
%conv_5_conv2d_readvariableop_resource*
&conv_5_biasadd_readvariableop_resource1
-batch_normalization_4_readvariableop_resource3
/batch_normalization_4_readvariableop_1_resourceB
>batch_normalization_4_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource7
3upsample_6_conv2d_transpose_readvariableop_resource.
*upsample_6_biasadd_readvariableop_resource)
%conv_6_conv2d_readvariableop_resource*
&conv_6_biasadd_readvariableop_resource1
-batch_normalization_5_readvariableop_resource3
/batch_normalization_5_readvariableop_1_resourceB
>batch_normalization_5_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource7
3upsample_7_conv2d_transpose_readvariableop_resource.
*upsample_7_biasadd_readvariableop_resource)
%conv_7_conv2d_readvariableop_resource*
&conv_7_biasadd_readvariableop_resource1
-batch_normalization_6_readvariableop_resource3
/batch_normalization_6_readvariableop_1_resourceB
>batch_normalization_6_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource7
3upsample_8_conv2d_transpose_readvariableop_resource.
*upsample_8_biasadd_readvariableop_resource)
%conv_8_conv2d_readvariableop_resource*
&conv_8_biasadd_readvariableop_resource1
-batch_normalization_7_readvariableop_resource3
/batch_normalization_7_readvariableop_1_resourceB
>batch_normalization_7_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource7
3upsample_9_conv2d_transpose_readvariableop_resource.
*upsample_9_biasadd_readvariableop_resource)
%conv_9_conv2d_readvariableop_resource*
&conv_9_biasadd_readvariableop_resource1
-batch_normalization_8_readvariableop_resource3
/batch_normalization_8_readvariableop_1_resourceB
>batch_normalization_8_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource(
$final_conv2d_readvariableop_resource)
%final_biasadd_readvariableop_resource
identity’"batch_normalization/AssignNewValue’$batch_normalization/AssignNewValue_1’$batch_normalization_1/AssignNewValue’&batch_normalization_1/AssignNewValue_1’$batch_normalization_2/AssignNewValue’&batch_normalization_2/AssignNewValue_1’$batch_normalization_3/AssignNewValue’&batch_normalization_3/AssignNewValue_1’$batch_normalization_4/AssignNewValue’&batch_normalization_4/AssignNewValue_1’$batch_normalization_5/AssignNewValue’&batch_normalization_5/AssignNewValue_1’$batch_normalization_6/AssignNewValue’&batch_normalization_6/AssignNewValue_1’$batch_normalization_7/AssignNewValue’&batch_normalization_7/AssignNewValue_1’$batch_normalization_8/AssignNewValue’&batch_normalization_8/AssignNewValue_1
tf_op_layer_RealDiv/RealDiv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?B2
tf_op_layer_RealDiv/RealDiv/yΏ
tf_op_layer_RealDiv/RealDivRealDivinputs&tf_op_layer_RealDiv/RealDiv/y:output:0*
T0*
_cloned(*0
_output_shapes
:?????????`ΐ2
tf_op_layer_RealDiv/RealDivs
tf_op_layer_Sub/Sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
tf_op_layer_Sub/Sub/yΌ
tf_op_layer_Sub/SubSubtf_op_layer_RealDiv/RealDiv:z:0tf_op_layer_Sub/Sub/y:output:0*
T0*
_cloned(*0
_output_shapes
:?????????`ΐ2
tf_op_layer_Sub/Subͺ
conv_1/Conv2D/ReadVariableOpReadVariableOp%conv_1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
conv_1/Conv2D/ReadVariableOpΚ
conv_1/Conv2DConv2Dtf_op_layer_Sub/Sub:z:0$conv_1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????`ΐ*
paddingSAME*
strides
2
conv_1/Conv2D‘
conv_1/BiasAdd/ReadVariableOpReadVariableOp&conv_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
conv_1/BiasAdd/ReadVariableOp₯
conv_1/BiasAddBiasAddconv_1/Conv2D:output:0%conv_1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????`ΐ2
conv_1/BiasAdd°
"batch_normalization/ReadVariableOpReadVariableOp+batch_normalization_readvariableop_resource*
_output_shapes
:*
dtype02$
"batch_normalization/ReadVariableOpΆ
$batch_normalization/ReadVariableOp_1ReadVariableOp-batch_normalization_readvariableop_1_resource*
_output_shapes
:*
dtype02&
$batch_normalization/ReadVariableOp_1γ
3batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOp<batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype025
3batch_normalization/FusedBatchNormV3/ReadVariableOpι
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp>batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype027
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1β
$batch_normalization/FusedBatchNormV3FusedBatchNormV3conv_1/BiasAdd:output:0*batch_normalization/ReadVariableOp:value:0,batch_normalization/ReadVariableOp_1:value:0;batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0=batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*L
_output_shapes:
8:?????????`ΐ:::::*
epsilon%o:*
exponential_avg_factor%
Χ#<2&
$batch_normalization/FusedBatchNormV3χ
"batch_normalization/AssignNewValueAssignVariableOp<batch_normalization_fusedbatchnormv3_readvariableop_resource1batch_normalization/FusedBatchNormV3:batch_mean:04^batch_normalization/FusedBatchNormV3/ReadVariableOp*O
_classE
CAloc:@batch_normalization/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02$
"batch_normalization/AssignNewValue
$batch_normalization/AssignNewValue_1AssignVariableOp>batch_normalization_fusedbatchnormv3_readvariableop_1_resource5batch_normalization/FusedBatchNormV3:batch_variance:06^batch_normalization/FusedBatchNormV3/ReadVariableOp_1*Q
_classG
ECloc:@batch_normalization/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02&
$batch_normalization/AssignNewValue_1

re_lu/ReluRelu(batch_normalization/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:?????????`ΐ2

re_lu/Reluΐ
max_pooling2d/MaxPoolMaxPoolre_lu/Relu:activations:0*/
_output_shapes
:?????????0`*
ksize
*
paddingVALID*
strides
2
max_pooling2d/MaxPoolͺ
conv_2/Conv2D/ReadVariableOpReadVariableOp%conv_2_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
conv_2/Conv2D/ReadVariableOpΠ
conv_2/Conv2DConv2Dmax_pooling2d/MaxPool:output:0$conv_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????0` *
paddingSAME*
strides
2
conv_2/Conv2D‘
conv_2/BiasAdd/ReadVariableOpReadVariableOp&conv_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02
conv_2/BiasAdd/ReadVariableOp€
conv_2/BiasAddBiasAddconv_2/Conv2D:output:0%conv_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????0` 2
conv_2/BiasAddΆ
$batch_normalization_1/ReadVariableOpReadVariableOp-batch_normalization_1_readvariableop_resource*
_output_shapes
: *
dtype02&
$batch_normalization_1/ReadVariableOpΌ
&batch_normalization_1/ReadVariableOp_1ReadVariableOp/batch_normalization_1_readvariableop_1_resource*
_output_shapes
: *
dtype02(
&batch_normalization_1/ReadVariableOp_1ι
5batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype027
5batch_normalization_1/FusedBatchNormV3/ReadVariableOpο
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype029
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ν
&batch_normalization_1/FusedBatchNormV3FusedBatchNormV3conv_2/BiasAdd:output:0,batch_normalization_1/ReadVariableOp:value:0.batch_normalization_1/ReadVariableOp_1:value:0=batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????0` : : : : :*
epsilon%o:*
exponential_avg_factor%
Χ#<2(
&batch_normalization_1/FusedBatchNormV3
$batch_normalization_1/AssignNewValueAssignVariableOp>batch_normalization_1_fusedbatchnormv3_readvariableop_resource3batch_normalization_1/FusedBatchNormV3:batch_mean:06^batch_normalization_1/FusedBatchNormV3/ReadVariableOp*Q
_classG
ECloc:@batch_normalization_1/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02&
$batch_normalization_1/AssignNewValue
&batch_normalization_1/AssignNewValue_1AssignVariableOp@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_1/FusedBatchNormV3:batch_variance:08^batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1*S
_classI
GEloc:@batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02(
&batch_normalization_1/AssignNewValue_1
re_lu_1/ReluRelu*batch_normalization_1/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????0` 2
re_lu_1/ReluΖ
max_pooling2d_1/MaxPoolMaxPoolre_lu_1/Relu:activations:0*/
_output_shapes
:?????????0 *
ksize
*
paddingVALID*
strides
2
max_pooling2d_1/MaxPoolͺ
conv_3/Conv2D/ReadVariableOpReadVariableOp%conv_3_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02
conv_3/Conv2D/ReadVariableOp?
conv_3/Conv2DConv2D max_pooling2d_1/MaxPool:output:0$conv_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????0@*
paddingSAME*
strides
2
conv_3/Conv2D‘
conv_3/BiasAdd/ReadVariableOpReadVariableOp&conv_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
conv_3/BiasAdd/ReadVariableOp€
conv_3/BiasAddBiasAddconv_3/Conv2D:output:0%conv_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????0@2
conv_3/BiasAddΆ
$batch_normalization_2/ReadVariableOpReadVariableOp-batch_normalization_2_readvariableop_resource*
_output_shapes
:@*
dtype02&
$batch_normalization_2/ReadVariableOpΌ
&batch_normalization_2/ReadVariableOp_1ReadVariableOp/batch_normalization_2_readvariableop_1_resource*
_output_shapes
:@*
dtype02(
&batch_normalization_2/ReadVariableOp_1ι
5batch_normalization_2/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype027
5batch_normalization_2/FusedBatchNormV3/ReadVariableOpο
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype029
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1ν
&batch_normalization_2/FusedBatchNormV3FusedBatchNormV3conv_3/BiasAdd:output:0,batch_normalization_2/ReadVariableOp:value:0.batch_normalization_2/ReadVariableOp_1:value:0=batch_normalization_2/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????0@:@:@:@:@:*
epsilon%o:*
exponential_avg_factor%
Χ#<2(
&batch_normalization_2/FusedBatchNormV3
$batch_normalization_2/AssignNewValueAssignVariableOp>batch_normalization_2_fusedbatchnormv3_readvariableop_resource3batch_normalization_2/FusedBatchNormV3:batch_mean:06^batch_normalization_2/FusedBatchNormV3/ReadVariableOp*Q
_classG
ECloc:@batch_normalization_2/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02&
$batch_normalization_2/AssignNewValue
&batch_normalization_2/AssignNewValue_1AssignVariableOp@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_2/FusedBatchNormV3:batch_variance:08^batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1*S
_classI
GEloc:@batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02(
&batch_normalization_2/AssignNewValue_1
re_lu_2/ReluRelu*batch_normalization_2/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????0@2
re_lu_2/ReluΖ
max_pooling2d_2/MaxPoolMaxPoolre_lu_2/Relu:activations:0*/
_output_shapes
:?????????@*
ksize
*
paddingVALID*
strides
2
max_pooling2d_2/MaxPool«
conv_4/Conv2D/ReadVariableOpReadVariableOp%conv_4_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype02
conv_4/Conv2D/ReadVariableOpΣ
conv_4/Conv2DConv2D max_pooling2d_2/MaxPool:output:0$conv_4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????*
paddingSAME*
strides
2
conv_4/Conv2D’
conv_4/BiasAdd/ReadVariableOpReadVariableOp&conv_4_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
conv_4/BiasAdd/ReadVariableOp₯
conv_4/BiasAddBiasAddconv_4/Conv2D:output:0%conv_4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????2
conv_4/BiasAdd·
$batch_normalization_3/ReadVariableOpReadVariableOp-batch_normalization_3_readvariableop_resource*
_output_shapes	
:*
dtype02&
$batch_normalization_3/ReadVariableOp½
&batch_normalization_3/ReadVariableOp_1ReadVariableOp/batch_normalization_3_readvariableop_1_resource*
_output_shapes	
:*
dtype02(
&batch_normalization_3/ReadVariableOp_1κ
5batch_normalization_3/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_3_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype027
5batch_normalization_3/FusedBatchNormV3/ReadVariableOpπ
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype029
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1ς
&batch_normalization_3/FusedBatchNormV3FusedBatchNormV3conv_4/BiasAdd:output:0,batch_normalization_3/ReadVariableOp:value:0.batch_normalization_3/ReadVariableOp_1:value:0=batch_normalization_3/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:?????????:::::*
epsilon%o:*
exponential_avg_factor%
Χ#<2(
&batch_normalization_3/FusedBatchNormV3
$batch_normalization_3/AssignNewValueAssignVariableOp>batch_normalization_3_fusedbatchnormv3_readvariableop_resource3batch_normalization_3/FusedBatchNormV3:batch_mean:06^batch_normalization_3/FusedBatchNormV3/ReadVariableOp*Q
_classG
ECloc:@batch_normalization_3/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02&
$batch_normalization_3/AssignNewValue
&batch_normalization_3/AssignNewValue_1AssignVariableOp@batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_3/FusedBatchNormV3:batch_variance:08^batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1*S
_classI
GEloc:@batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02(
&batch_normalization_3/AssignNewValue_1
re_lu_3/ReluRelu*batch_normalization_3/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:?????????2
re_lu_3/ReluΗ
max_pooling2d_3/MaxPoolMaxPoolre_lu_3/Relu:activations:0*0
_output_shapes
:?????????*
ksize
*
paddingVALID*
strides
2
max_pooling2d_3/MaxPool¬
conv_5/Conv2D/ReadVariableOpReadVariableOp%conv_5_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02
conv_5/Conv2D/ReadVariableOpΣ
conv_5/Conv2DConv2D max_pooling2d_3/MaxPool:output:0$conv_5/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????*
paddingSAME*
strides
2
conv_5/Conv2D’
conv_5/BiasAdd/ReadVariableOpReadVariableOp&conv_5_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
conv_5/BiasAdd/ReadVariableOp₯
conv_5/BiasAddBiasAddconv_5/Conv2D:output:0%conv_5/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????2
conv_5/BiasAdd·
$batch_normalization_4/ReadVariableOpReadVariableOp-batch_normalization_4_readvariableop_resource*
_output_shapes	
:*
dtype02&
$batch_normalization_4/ReadVariableOp½
&batch_normalization_4/ReadVariableOp_1ReadVariableOp/batch_normalization_4_readvariableop_1_resource*
_output_shapes	
:*
dtype02(
&batch_normalization_4/ReadVariableOp_1κ
5batch_normalization_4/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_4_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype027
5batch_normalization_4/FusedBatchNormV3/ReadVariableOpπ
7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype029
7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1ς
&batch_normalization_4/FusedBatchNormV3FusedBatchNormV3conv_5/BiasAdd:output:0,batch_normalization_4/ReadVariableOp:value:0.batch_normalization_4/ReadVariableOp_1:value:0=batch_normalization_4/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:?????????:::::*
epsilon%o:*
exponential_avg_factor%
Χ#<2(
&batch_normalization_4/FusedBatchNormV3
$batch_normalization_4/AssignNewValueAssignVariableOp>batch_normalization_4_fusedbatchnormv3_readvariableop_resource3batch_normalization_4/FusedBatchNormV3:batch_mean:06^batch_normalization_4/FusedBatchNormV3/ReadVariableOp*Q
_classG
ECloc:@batch_normalization_4/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02&
$batch_normalization_4/AssignNewValue
&batch_normalization_4/AssignNewValue_1AssignVariableOp@batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_4/FusedBatchNormV3:batch_variance:08^batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1*S
_classI
GEloc:@batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02(
&batch_normalization_4/AssignNewValue_1
re_lu_4/ReluRelu*batch_normalization_4/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:?????????2
re_lu_4/Relun
upsample_6/ShapeShapere_lu_4/Relu:activations:0*
T0*
_output_shapes
:2
upsample_6/Shape
upsample_6/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
upsample_6/strided_slice/stack
 upsample_6/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 upsample_6/strided_slice/stack_1
 upsample_6/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 upsample_6/strided_slice/stack_2€
upsample_6/strided_sliceStridedSliceupsample_6/Shape:output:0'upsample_6/strided_slice/stack:output:0)upsample_6/strided_slice/stack_1:output:0)upsample_6/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
upsample_6/strided_slicej
upsample_6/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2
upsample_6/stack/1j
upsample_6/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2
upsample_6/stack/2k
upsample_6/stack/3Const*
_output_shapes
: *
dtype0*
value
B :2
upsample_6/stack/3Τ
upsample_6/stackPack!upsample_6/strided_slice:output:0upsample_6/stack/1:output:0upsample_6/stack/2:output:0upsample_6/stack/3:output:0*
N*
T0*
_output_shapes
:2
upsample_6/stack
 upsample_6/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2"
 upsample_6/strided_slice_1/stack
"upsample_6/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2$
"upsample_6/strided_slice_1/stack_1
"upsample_6/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"upsample_6/strided_slice_1/stack_2?
upsample_6/strided_slice_1StridedSliceupsample_6/stack:output:0)upsample_6/strided_slice_1/stack:output:0+upsample_6/strided_slice_1/stack_1:output:0+upsample_6/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
upsample_6/strided_slice_1Φ
*upsample_6/conv2d_transpose/ReadVariableOpReadVariableOp3upsample_6_conv2d_transpose_readvariableop_resource*(
_output_shapes
:*
dtype02,
*upsample_6/conv2d_transpose/ReadVariableOp 
upsample_6/conv2d_transposeConv2DBackpropInputupsample_6/stack:output:02upsample_6/conv2d_transpose/ReadVariableOp:value:0re_lu_4/Relu:activations:0*
T0*0
_output_shapes
:?????????*
paddingVALID*
strides
2
upsample_6/conv2d_transpose?
!upsample_6/BiasAdd/ReadVariableOpReadVariableOp*upsample_6_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02#
!upsample_6/BiasAdd/ReadVariableOpΏ
upsample_6/BiasAddBiasAdd$upsample_6/conv2d_transpose:output:0)upsample_6/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????2
upsample_6/BiasAdd
"tf_op_layer_concat_6/concat_6/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2$
"tf_op_layer_concat_6/concat_6/axis
tf_op_layer_concat_6/concat_6ConcatV2upsample_6/BiasAdd:output:0re_lu_3/Relu:activations:0+tf_op_layer_concat_6/concat_6/axis:output:0*
N*
T0*
_cloned(*0
_output_shapes
:?????????2
tf_op_layer_concat_6/concat_6¬
conv_6/Conv2D/ReadVariableOpReadVariableOp%conv_6_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02
conv_6/Conv2D/ReadVariableOpΩ
conv_6/Conv2DConv2D&tf_op_layer_concat_6/concat_6:output:0$conv_6/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????*
paddingSAME*
strides
2
conv_6/Conv2D’
conv_6/BiasAdd/ReadVariableOpReadVariableOp&conv_6_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
conv_6/BiasAdd/ReadVariableOp₯
conv_6/BiasAddBiasAddconv_6/Conv2D:output:0%conv_6/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????2
conv_6/BiasAdd·
$batch_normalization_5/ReadVariableOpReadVariableOp-batch_normalization_5_readvariableop_resource*
_output_shapes	
:*
dtype02&
$batch_normalization_5/ReadVariableOp½
&batch_normalization_5/ReadVariableOp_1ReadVariableOp/batch_normalization_5_readvariableop_1_resource*
_output_shapes	
:*
dtype02(
&batch_normalization_5/ReadVariableOp_1κ
5batch_normalization_5/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_5_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype027
5batch_normalization_5/FusedBatchNormV3/ReadVariableOpπ
7batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype029
7batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1ς
&batch_normalization_5/FusedBatchNormV3FusedBatchNormV3conv_6/BiasAdd:output:0,batch_normalization_5/ReadVariableOp:value:0.batch_normalization_5/ReadVariableOp_1:value:0=batch_normalization_5/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:?????????:::::*
epsilon%o:*
exponential_avg_factor%
Χ#<2(
&batch_normalization_5/FusedBatchNormV3
$batch_normalization_5/AssignNewValueAssignVariableOp>batch_normalization_5_fusedbatchnormv3_readvariableop_resource3batch_normalization_5/FusedBatchNormV3:batch_mean:06^batch_normalization_5/FusedBatchNormV3/ReadVariableOp*Q
_classG
ECloc:@batch_normalization_5/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02&
$batch_normalization_5/AssignNewValue
&batch_normalization_5/AssignNewValue_1AssignVariableOp@batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_5/FusedBatchNormV3:batch_variance:08^batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1*S
_classI
GEloc:@batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02(
&batch_normalization_5/AssignNewValue_1
re_lu_5/ReluRelu*batch_normalization_5/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:?????????2
re_lu_5/Relun
upsample_7/ShapeShapere_lu_5/Relu:activations:0*
T0*
_output_shapes
:2
upsample_7/Shape
upsample_7/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
upsample_7/strided_slice/stack
 upsample_7/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 upsample_7/strided_slice/stack_1
 upsample_7/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 upsample_7/strided_slice/stack_2€
upsample_7/strided_sliceStridedSliceupsample_7/Shape:output:0'upsample_7/strided_slice/stack:output:0)upsample_7/strided_slice/stack_1:output:0)upsample_7/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
upsample_7/strided_slicej
upsample_7/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2
upsample_7/stack/1j
upsample_7/stack/2Const*
_output_shapes
: *
dtype0*
value	B :02
upsample_7/stack/2j
upsample_7/stack/3Const*
_output_shapes
: *
dtype0*
value	B :@2
upsample_7/stack/3Τ
upsample_7/stackPack!upsample_7/strided_slice:output:0upsample_7/stack/1:output:0upsample_7/stack/2:output:0upsample_7/stack/3:output:0*
N*
T0*
_output_shapes
:2
upsample_7/stack
 upsample_7/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2"
 upsample_7/strided_slice_1/stack
"upsample_7/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2$
"upsample_7/strided_slice_1/stack_1
"upsample_7/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"upsample_7/strided_slice_1/stack_2?
upsample_7/strided_slice_1StridedSliceupsample_7/stack:output:0)upsample_7/strided_slice_1/stack:output:0+upsample_7/strided_slice_1/stack_1:output:0+upsample_7/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
upsample_7/strided_slice_1Υ
*upsample_7/conv2d_transpose/ReadVariableOpReadVariableOp3upsample_7_conv2d_transpose_readvariableop_resource*'
_output_shapes
:@*
dtype02,
*upsample_7/conv2d_transpose/ReadVariableOp
upsample_7/conv2d_transposeConv2DBackpropInputupsample_7/stack:output:02upsample_7/conv2d_transpose/ReadVariableOp:value:0re_lu_5/Relu:activations:0*
T0*/
_output_shapes
:?????????0@*
paddingVALID*
strides
2
upsample_7/conv2d_transpose­
!upsample_7/BiasAdd/ReadVariableOpReadVariableOp*upsample_7_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02#
!upsample_7/BiasAdd/ReadVariableOpΎ
upsample_7/BiasAddBiasAdd$upsample_7/conv2d_transpose:output:0)upsample_7/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????0@2
upsample_7/BiasAdd
"tf_op_layer_concat_7/concat_7/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2$
"tf_op_layer_concat_7/concat_7/axis
tf_op_layer_concat_7/concat_7ConcatV2upsample_7/BiasAdd:output:0re_lu_2/Relu:activations:0+tf_op_layer_concat_7/concat_7/axis:output:0*
N*
T0*
_cloned(*0
_output_shapes
:?????????02
tf_op_layer_concat_7/concat_7«
conv_7/Conv2D/ReadVariableOpReadVariableOp%conv_7_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype02
conv_7/Conv2D/ReadVariableOpΨ
conv_7/Conv2DConv2D&tf_op_layer_concat_7/concat_7:output:0$conv_7/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????0@*
paddingSAME*
strides
2
conv_7/Conv2D‘
conv_7/BiasAdd/ReadVariableOpReadVariableOp&conv_7_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
conv_7/BiasAdd/ReadVariableOp€
conv_7/BiasAddBiasAddconv_7/Conv2D:output:0%conv_7/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????0@2
conv_7/BiasAddΆ
$batch_normalization_6/ReadVariableOpReadVariableOp-batch_normalization_6_readvariableop_resource*
_output_shapes
:@*
dtype02&
$batch_normalization_6/ReadVariableOpΌ
&batch_normalization_6/ReadVariableOp_1ReadVariableOp/batch_normalization_6_readvariableop_1_resource*
_output_shapes
:@*
dtype02(
&batch_normalization_6/ReadVariableOp_1ι
5batch_normalization_6/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_6_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype027
5batch_normalization_6/FusedBatchNormV3/ReadVariableOpο
7batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype029
7batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1ν
&batch_normalization_6/FusedBatchNormV3FusedBatchNormV3conv_7/BiasAdd:output:0,batch_normalization_6/ReadVariableOp:value:0.batch_normalization_6/ReadVariableOp_1:value:0=batch_normalization_6/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????0@:@:@:@:@:*
epsilon%o:*
exponential_avg_factor%
Χ#<2(
&batch_normalization_6/FusedBatchNormV3
$batch_normalization_6/AssignNewValueAssignVariableOp>batch_normalization_6_fusedbatchnormv3_readvariableop_resource3batch_normalization_6/FusedBatchNormV3:batch_mean:06^batch_normalization_6/FusedBatchNormV3/ReadVariableOp*Q
_classG
ECloc:@batch_normalization_6/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02&
$batch_normalization_6/AssignNewValue
&batch_normalization_6/AssignNewValue_1AssignVariableOp@batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_6/FusedBatchNormV3:batch_variance:08^batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1*S
_classI
GEloc:@batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02(
&batch_normalization_6/AssignNewValue_1
re_lu_6/ReluRelu*batch_normalization_6/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????0@2
re_lu_6/Relun
upsample_8/ShapeShapere_lu_6/Relu:activations:0*
T0*
_output_shapes
:2
upsample_8/Shape
upsample_8/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
upsample_8/strided_slice/stack
 upsample_8/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 upsample_8/strided_slice/stack_1
 upsample_8/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 upsample_8/strided_slice/stack_2€
upsample_8/strided_sliceStridedSliceupsample_8/Shape:output:0'upsample_8/strided_slice/stack:output:0)upsample_8/strided_slice/stack_1:output:0)upsample_8/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
upsample_8/strided_slicej
upsample_8/stack/1Const*
_output_shapes
: *
dtype0*
value	B :02
upsample_8/stack/1j
upsample_8/stack/2Const*
_output_shapes
: *
dtype0*
value	B :`2
upsample_8/stack/2j
upsample_8/stack/3Const*
_output_shapes
: *
dtype0*
value	B : 2
upsample_8/stack/3Τ
upsample_8/stackPack!upsample_8/strided_slice:output:0upsample_8/stack/1:output:0upsample_8/stack/2:output:0upsample_8/stack/3:output:0*
N*
T0*
_output_shapes
:2
upsample_8/stack
 upsample_8/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2"
 upsample_8/strided_slice_1/stack
"upsample_8/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2$
"upsample_8/strided_slice_1/stack_1
"upsample_8/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"upsample_8/strided_slice_1/stack_2?
upsample_8/strided_slice_1StridedSliceupsample_8/stack:output:0)upsample_8/strided_slice_1/stack:output:0+upsample_8/strided_slice_1/stack_1:output:0+upsample_8/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
upsample_8/strided_slice_1Τ
*upsample_8/conv2d_transpose/ReadVariableOpReadVariableOp3upsample_8_conv2d_transpose_readvariableop_resource*&
_output_shapes
: @*
dtype02,
*upsample_8/conv2d_transpose/ReadVariableOp
upsample_8/conv2d_transposeConv2DBackpropInputupsample_8/stack:output:02upsample_8/conv2d_transpose/ReadVariableOp:value:0re_lu_6/Relu:activations:0*
T0*/
_output_shapes
:?????????0` *
paddingVALID*
strides
2
upsample_8/conv2d_transpose­
!upsample_8/BiasAdd/ReadVariableOpReadVariableOp*upsample_8_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02#
!upsample_8/BiasAdd/ReadVariableOpΎ
upsample_8/BiasAddBiasAdd$upsample_8/conv2d_transpose:output:0)upsample_8/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????0` 2
upsample_8/BiasAdd
"tf_op_layer_concat_8/concat_8/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2$
"tf_op_layer_concat_8/concat_8/axis
tf_op_layer_concat_8/concat_8ConcatV2upsample_8/BiasAdd:output:0re_lu_1/Relu:activations:0+tf_op_layer_concat_8/concat_8/axis:output:0*
N*
T0*
_cloned(*/
_output_shapes
:?????????0`@2
tf_op_layer_concat_8/concat_8ͺ
conv_8/Conv2D/ReadVariableOpReadVariableOp%conv_8_conv2d_readvariableop_resource*&
_output_shapes
:@ *
dtype02
conv_8/Conv2D/ReadVariableOpΨ
conv_8/Conv2DConv2D&tf_op_layer_concat_8/concat_8:output:0$conv_8/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????0` *
paddingSAME*
strides
2
conv_8/Conv2D‘
conv_8/BiasAdd/ReadVariableOpReadVariableOp&conv_8_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02
conv_8/BiasAdd/ReadVariableOp€
conv_8/BiasAddBiasAddconv_8/Conv2D:output:0%conv_8/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????0` 2
conv_8/BiasAddΆ
$batch_normalization_7/ReadVariableOpReadVariableOp-batch_normalization_7_readvariableop_resource*
_output_shapes
: *
dtype02&
$batch_normalization_7/ReadVariableOpΌ
&batch_normalization_7/ReadVariableOp_1ReadVariableOp/batch_normalization_7_readvariableop_1_resource*
_output_shapes
: *
dtype02(
&batch_normalization_7/ReadVariableOp_1ι
5batch_normalization_7/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_7_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype027
5batch_normalization_7/FusedBatchNormV3/ReadVariableOpο
7batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype029
7batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1ν
&batch_normalization_7/FusedBatchNormV3FusedBatchNormV3conv_8/BiasAdd:output:0,batch_normalization_7/ReadVariableOp:value:0.batch_normalization_7/ReadVariableOp_1:value:0=batch_normalization_7/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????0` : : : : :*
epsilon%o:*
exponential_avg_factor%
Χ#<2(
&batch_normalization_7/FusedBatchNormV3
$batch_normalization_7/AssignNewValueAssignVariableOp>batch_normalization_7_fusedbatchnormv3_readvariableop_resource3batch_normalization_7/FusedBatchNormV3:batch_mean:06^batch_normalization_7/FusedBatchNormV3/ReadVariableOp*Q
_classG
ECloc:@batch_normalization_7/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02&
$batch_normalization_7/AssignNewValue
&batch_normalization_7/AssignNewValue_1AssignVariableOp@batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_7/FusedBatchNormV3:batch_variance:08^batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1*S
_classI
GEloc:@batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02(
&batch_normalization_7/AssignNewValue_1
re_lu_7/ReluRelu*batch_normalization_7/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????0` 2
re_lu_7/Relun
upsample_9/ShapeShapere_lu_7/Relu:activations:0*
T0*
_output_shapes
:2
upsample_9/Shape
upsample_9/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
upsample_9/strided_slice/stack
 upsample_9/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 upsample_9/strided_slice/stack_1
 upsample_9/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 upsample_9/strided_slice/stack_2€
upsample_9/strided_sliceStridedSliceupsample_9/Shape:output:0'upsample_9/strided_slice/stack:output:0)upsample_9/strided_slice/stack_1:output:0)upsample_9/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
upsample_9/strided_slicej
upsample_9/stack/1Const*
_output_shapes
: *
dtype0*
value	B :`2
upsample_9/stack/1k
upsample_9/stack/2Const*
_output_shapes
: *
dtype0*
value
B :ΐ2
upsample_9/stack/2j
upsample_9/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2
upsample_9/stack/3Τ
upsample_9/stackPack!upsample_9/strided_slice:output:0upsample_9/stack/1:output:0upsample_9/stack/2:output:0upsample_9/stack/3:output:0*
N*
T0*
_output_shapes
:2
upsample_9/stack
 upsample_9/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2"
 upsample_9/strided_slice_1/stack
"upsample_9/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2$
"upsample_9/strided_slice_1/stack_1
"upsample_9/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"upsample_9/strided_slice_1/stack_2?
upsample_9/strided_slice_1StridedSliceupsample_9/stack:output:0)upsample_9/strided_slice_1/stack:output:0+upsample_9/strided_slice_1/stack_1:output:0+upsample_9/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
upsample_9/strided_slice_1Τ
*upsample_9/conv2d_transpose/ReadVariableOpReadVariableOp3upsample_9_conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype02,
*upsample_9/conv2d_transpose/ReadVariableOp 
upsample_9/conv2d_transposeConv2DBackpropInputupsample_9/stack:output:02upsample_9/conv2d_transpose/ReadVariableOp:value:0re_lu_7/Relu:activations:0*
T0*0
_output_shapes
:?????????`ΐ*
paddingVALID*
strides
2
upsample_9/conv2d_transpose­
!upsample_9/BiasAdd/ReadVariableOpReadVariableOp*upsample_9_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02#
!upsample_9/BiasAdd/ReadVariableOpΏ
upsample_9/BiasAddBiasAdd$upsample_9/conv2d_transpose:output:0)upsample_9/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????`ΐ2
upsample_9/BiasAdd
"tf_op_layer_concat_9/concat_9/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2$
"tf_op_layer_concat_9/concat_9/axis
tf_op_layer_concat_9/concat_9ConcatV2upsample_9/BiasAdd:output:0re_lu/Relu:activations:0+tf_op_layer_concat_9/concat_9/axis:output:0*
N*
T0*
_cloned(*0
_output_shapes
:?????????`ΐ 2
tf_op_layer_concat_9/concat_9ͺ
conv_9/Conv2D/ReadVariableOpReadVariableOp%conv_9_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
conv_9/Conv2D/ReadVariableOpΩ
conv_9/Conv2DConv2D&tf_op_layer_concat_9/concat_9:output:0$conv_9/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????`ΐ*
paddingSAME*
strides
2
conv_9/Conv2D‘
conv_9/BiasAdd/ReadVariableOpReadVariableOp&conv_9_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
conv_9/BiasAdd/ReadVariableOp₯
conv_9/BiasAddBiasAddconv_9/Conv2D:output:0%conv_9/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????`ΐ2
conv_9/BiasAddΆ
$batch_normalization_8/ReadVariableOpReadVariableOp-batch_normalization_8_readvariableop_resource*
_output_shapes
:*
dtype02&
$batch_normalization_8/ReadVariableOpΌ
&batch_normalization_8/ReadVariableOp_1ReadVariableOp/batch_normalization_8_readvariableop_1_resource*
_output_shapes
:*
dtype02(
&batch_normalization_8/ReadVariableOp_1ι
5batch_normalization_8/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_8_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype027
5batch_normalization_8/FusedBatchNormV3/ReadVariableOpο
7batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype029
7batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1ξ
&batch_normalization_8/FusedBatchNormV3FusedBatchNormV3conv_9/BiasAdd:output:0,batch_normalization_8/ReadVariableOp:value:0.batch_normalization_8/ReadVariableOp_1:value:0=batch_normalization_8/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*L
_output_shapes:
8:?????????`ΐ:::::*
epsilon%o:*
exponential_avg_factor%
Χ#<2(
&batch_normalization_8/FusedBatchNormV3
$batch_normalization_8/AssignNewValueAssignVariableOp>batch_normalization_8_fusedbatchnormv3_readvariableop_resource3batch_normalization_8/FusedBatchNormV3:batch_mean:06^batch_normalization_8/FusedBatchNormV3/ReadVariableOp*Q
_classG
ECloc:@batch_normalization_8/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02&
$batch_normalization_8/AssignNewValue
&batch_normalization_8/AssignNewValue_1AssignVariableOp@batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_8/FusedBatchNormV3:batch_variance:08^batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1*S
_classI
GEloc:@batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02(
&batch_normalization_8/AssignNewValue_1
re_lu_8/ReluRelu*batch_normalization_8/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:?????????`ΐ2
re_lu_8/Relu§
final/Conv2D/ReadVariableOpReadVariableOp$final_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
final/Conv2D/ReadVariableOpΚ
final/Conv2DConv2Dre_lu_8/Relu:activations:0#final/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????`ΐ*
paddingSAME*
strides
2
final/Conv2D
final/BiasAdd/ReadVariableOpReadVariableOp%final_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
final/BiasAdd/ReadVariableOp‘
final/BiasAddBiasAddfinal/Conv2D:output:0$final/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????`ΐ2
final/BiasAddΏ
IdentityIdentityfinal/BiasAdd:output:0#^batch_normalization/AssignNewValue%^batch_normalization/AssignNewValue_1%^batch_normalization_1/AssignNewValue'^batch_normalization_1/AssignNewValue_1%^batch_normalization_2/AssignNewValue'^batch_normalization_2/AssignNewValue_1%^batch_normalization_3/AssignNewValue'^batch_normalization_3/AssignNewValue_1%^batch_normalization_4/AssignNewValue'^batch_normalization_4/AssignNewValue_1%^batch_normalization_5/AssignNewValue'^batch_normalization_5/AssignNewValue_1%^batch_normalization_6/AssignNewValue'^batch_normalization_6/AssignNewValue_1%^batch_normalization_7/AssignNewValue'^batch_normalization_7/AssignNewValue_1%^batch_normalization_8/AssignNewValue'^batch_normalization_8/AssignNewValue_1*
T0*0
_output_shapes
:?????????`ΐ2

Identity"
identityIdentity:output:0*±
_input_shapes
:?????????`ΐ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::2H
"batch_normalization/AssignNewValue"batch_normalization/AssignNewValue2L
$batch_normalization/AssignNewValue_1$batch_normalization/AssignNewValue_12L
$batch_normalization_1/AssignNewValue$batch_normalization_1/AssignNewValue2P
&batch_normalization_1/AssignNewValue_1&batch_normalization_1/AssignNewValue_12L
$batch_normalization_2/AssignNewValue$batch_normalization_2/AssignNewValue2P
&batch_normalization_2/AssignNewValue_1&batch_normalization_2/AssignNewValue_12L
$batch_normalization_3/AssignNewValue$batch_normalization_3/AssignNewValue2P
&batch_normalization_3/AssignNewValue_1&batch_normalization_3/AssignNewValue_12L
$batch_normalization_4/AssignNewValue$batch_normalization_4/AssignNewValue2P
&batch_normalization_4/AssignNewValue_1&batch_normalization_4/AssignNewValue_12L
$batch_normalization_5/AssignNewValue$batch_normalization_5/AssignNewValue2P
&batch_normalization_5/AssignNewValue_1&batch_normalization_5/AssignNewValue_12L
$batch_normalization_6/AssignNewValue$batch_normalization_6/AssignNewValue2P
&batch_normalization_6/AssignNewValue_1&batch_normalization_6/AssignNewValue_12L
$batch_normalization_7/AssignNewValue$batch_normalization_7/AssignNewValue2P
&batch_normalization_7/AssignNewValue_1&batch_normalization_7/AssignNewValue_12L
$batch_normalization_8/AssignNewValue$batch_normalization_8/AssignNewValue2P
&batch_normalization_8/AssignNewValue_1&batch_normalization_8/AssignNewValue_1:X T
0
_output_shapes
:?????????`ΐ
 
_user_specified_nameinputs

°
S__inference_batch_normalization_1_layer_call_and_return_conditional_losses_57378445

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity’AssignNewValue’AssignNewValue_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ψ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????0` : : : : :*
epsilon%o:*
exponential_avg_factor%
Χ#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*/
_output_shapes
:?????????0` 2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????0` ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:W S
/
_output_shapes
:?????????0` 
 
_user_specified_nameinputs

i
M__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_57377537

inputs
identity­
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
2	
MaxPool
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
Ε
i
M__inference_tf_op_layer_Sub_layer_call_and_return_conditional_losses_57381217

inputs
identityS
Sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
Sub/ys
SubSubinputsSub/y:output:0*
T0*
_cloned(*0
_output_shapes
:?????????`ΐ2
Subd
IdentityIdentitySub:z:0*
T0*0
_output_shapes
:?????????`ΐ2

Identity"
identityIdentity:output:0*/
_input_shapes
:?????????`ΐ:X T
0
_output_shapes
:?????????`ΐ
 
_user_specified_nameinputs
Λ
ί
/__inference_functional_1_layer_call_fn_57380199

imageinput
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

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28

unknown_29

unknown_30

unknown_31

unknown_32

unknown_33

unknown_34

unknown_35

unknown_36

unknown_37

unknown_38

unknown_39

unknown_40

unknown_41

unknown_42

unknown_43

unknown_44

unknown_45

unknown_46

unknown_47

unknown_48

unknown_49

unknown_50

unknown_51

unknown_52

unknown_53

unknown_54

unknown_55

unknown_56

unknown_57

unknown_58

unknown_59

unknown_60

unknown_61

unknown_62
identity’StatefulPartitionedCallη	
StatefulPartitionedCallStatefulPartitionedCall
imageinputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46
unknown_47
unknown_48
unknown_49
unknown_50
unknown_51
unknown_52
unknown_53
unknown_54
unknown_55
unknown_56
unknown_57
unknown_58
unknown_59
unknown_60
unknown_61
unknown_62*L
TinE
C2A*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????`ΐ*b
_read_only_resource_inputsD
B@	
 !"#$%&'()*+,-./0123456789:;<=>?@*2
config_proto" 

CPU

GPU2 *0J 8 *S
fNRL
J__inference_functional_1_layer_call_and_return_conditional_losses_573800682
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:?????????`ΐ2

Identity"
identityIdentity:output:0*±
_input_shapes
:?????????`ΐ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:\ X
0
_output_shapes
:?????????`ΐ
$
_user_specified_name
imageInput
Ι
?
Q__inference_batch_normalization_layer_call_and_return_conditional_losses_57377141

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity’AssignNewValue’AssignNewValue_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1κ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o:*
exponential_avg_factor%
Χ#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1¦
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
ρω
"
#__inference__wrapped_model_57377079

imageinput6
2functional_1_conv_1_conv2d_readvariableop_resource7
3functional_1_conv_1_biasadd_readvariableop_resource<
8functional_1_batch_normalization_readvariableop_resource>
:functional_1_batch_normalization_readvariableop_1_resourceM
Ifunctional_1_batch_normalization_fusedbatchnormv3_readvariableop_resourceO
Kfunctional_1_batch_normalization_fusedbatchnormv3_readvariableop_1_resource6
2functional_1_conv_2_conv2d_readvariableop_resource7
3functional_1_conv_2_biasadd_readvariableop_resource>
:functional_1_batch_normalization_1_readvariableop_resource@
<functional_1_batch_normalization_1_readvariableop_1_resourceO
Kfunctional_1_batch_normalization_1_fusedbatchnormv3_readvariableop_resourceQ
Mfunctional_1_batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource6
2functional_1_conv_3_conv2d_readvariableop_resource7
3functional_1_conv_3_biasadd_readvariableop_resource>
:functional_1_batch_normalization_2_readvariableop_resource@
<functional_1_batch_normalization_2_readvariableop_1_resourceO
Kfunctional_1_batch_normalization_2_fusedbatchnormv3_readvariableop_resourceQ
Mfunctional_1_batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource6
2functional_1_conv_4_conv2d_readvariableop_resource7
3functional_1_conv_4_biasadd_readvariableop_resource>
:functional_1_batch_normalization_3_readvariableop_resource@
<functional_1_batch_normalization_3_readvariableop_1_resourceO
Kfunctional_1_batch_normalization_3_fusedbatchnormv3_readvariableop_resourceQ
Mfunctional_1_batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource6
2functional_1_conv_5_conv2d_readvariableop_resource7
3functional_1_conv_5_biasadd_readvariableop_resource>
:functional_1_batch_normalization_4_readvariableop_resource@
<functional_1_batch_normalization_4_readvariableop_1_resourceO
Kfunctional_1_batch_normalization_4_fusedbatchnormv3_readvariableop_resourceQ
Mfunctional_1_batch_normalization_4_fusedbatchnormv3_readvariableop_1_resourceD
@functional_1_upsample_6_conv2d_transpose_readvariableop_resource;
7functional_1_upsample_6_biasadd_readvariableop_resource6
2functional_1_conv_6_conv2d_readvariableop_resource7
3functional_1_conv_6_biasadd_readvariableop_resource>
:functional_1_batch_normalization_5_readvariableop_resource@
<functional_1_batch_normalization_5_readvariableop_1_resourceO
Kfunctional_1_batch_normalization_5_fusedbatchnormv3_readvariableop_resourceQ
Mfunctional_1_batch_normalization_5_fusedbatchnormv3_readvariableop_1_resourceD
@functional_1_upsample_7_conv2d_transpose_readvariableop_resource;
7functional_1_upsample_7_biasadd_readvariableop_resource6
2functional_1_conv_7_conv2d_readvariableop_resource7
3functional_1_conv_7_biasadd_readvariableop_resource>
:functional_1_batch_normalization_6_readvariableop_resource@
<functional_1_batch_normalization_6_readvariableop_1_resourceO
Kfunctional_1_batch_normalization_6_fusedbatchnormv3_readvariableop_resourceQ
Mfunctional_1_batch_normalization_6_fusedbatchnormv3_readvariableop_1_resourceD
@functional_1_upsample_8_conv2d_transpose_readvariableop_resource;
7functional_1_upsample_8_biasadd_readvariableop_resource6
2functional_1_conv_8_conv2d_readvariableop_resource7
3functional_1_conv_8_biasadd_readvariableop_resource>
:functional_1_batch_normalization_7_readvariableop_resource@
<functional_1_batch_normalization_7_readvariableop_1_resourceO
Kfunctional_1_batch_normalization_7_fusedbatchnormv3_readvariableop_resourceQ
Mfunctional_1_batch_normalization_7_fusedbatchnormv3_readvariableop_1_resourceD
@functional_1_upsample_9_conv2d_transpose_readvariableop_resource;
7functional_1_upsample_9_biasadd_readvariableop_resource6
2functional_1_conv_9_conv2d_readvariableop_resource7
3functional_1_conv_9_biasadd_readvariableop_resource>
:functional_1_batch_normalization_8_readvariableop_resource@
<functional_1_batch_normalization_8_readvariableop_1_resourceO
Kfunctional_1_batch_normalization_8_fusedbatchnormv3_readvariableop_resourceQ
Mfunctional_1_batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource5
1functional_1_final_conv2d_readvariableop_resource6
2functional_1_final_biasadd_readvariableop_resource
identity
*functional_1/tf_op_layer_RealDiv/RealDiv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?B2,
*functional_1/tf_op_layer_RealDiv/RealDiv/yκ
(functional_1/tf_op_layer_RealDiv/RealDivRealDiv
imageinput3functional_1/tf_op_layer_RealDiv/RealDiv/y:output:0*
T0*
_cloned(*0
_output_shapes
:?????????`ΐ2*
(functional_1/tf_op_layer_RealDiv/RealDiv
"functional_1/tf_op_layer_Sub/Sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2$
"functional_1/tf_op_layer_Sub/Sub/yπ
 functional_1/tf_op_layer_Sub/SubSub,functional_1/tf_op_layer_RealDiv/RealDiv:z:0+functional_1/tf_op_layer_Sub/Sub/y:output:0*
T0*
_cloned(*0
_output_shapes
:?????????`ΐ2"
 functional_1/tf_op_layer_Sub/SubΡ
)functional_1/conv_1/Conv2D/ReadVariableOpReadVariableOp2functional_1_conv_1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02+
)functional_1/conv_1/Conv2D/ReadVariableOpώ
functional_1/conv_1/Conv2DConv2D$functional_1/tf_op_layer_Sub/Sub:z:01functional_1/conv_1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????`ΐ*
paddingSAME*
strides
2
functional_1/conv_1/Conv2DΘ
*functional_1/conv_1/BiasAdd/ReadVariableOpReadVariableOp3functional_1_conv_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02,
*functional_1/conv_1/BiasAdd/ReadVariableOpΩ
functional_1/conv_1/BiasAddBiasAdd#functional_1/conv_1/Conv2D:output:02functional_1/conv_1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????`ΐ2
functional_1/conv_1/BiasAddΧ
/functional_1/batch_normalization/ReadVariableOpReadVariableOp8functional_1_batch_normalization_readvariableop_resource*
_output_shapes
:*
dtype021
/functional_1/batch_normalization/ReadVariableOpέ
1functional_1/batch_normalization/ReadVariableOp_1ReadVariableOp:functional_1_batch_normalization_readvariableop_1_resource*
_output_shapes
:*
dtype023
1functional_1/batch_normalization/ReadVariableOp_1
@functional_1/batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOpIfunctional_1_batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02B
@functional_1/batch_normalization/FusedBatchNormV3/ReadVariableOp
Bfunctional_1/batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpKfunctional_1_batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02D
Bfunctional_1/batch_normalization/FusedBatchNormV3/ReadVariableOp_1―
1functional_1/batch_normalization/FusedBatchNormV3FusedBatchNormV3$functional_1/conv_1/BiasAdd:output:07functional_1/batch_normalization/ReadVariableOp:value:09functional_1/batch_normalization/ReadVariableOp_1:value:0Hfunctional_1/batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0Jfunctional_1/batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*L
_output_shapes:
8:?????????`ΐ:::::*
epsilon%o:*
is_training( 23
1functional_1/batch_normalization/FusedBatchNormV3¬
functional_1/re_lu/ReluRelu5functional_1/batch_normalization/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:?????????`ΐ2
functional_1/re_lu/Reluη
"functional_1/max_pooling2d/MaxPoolMaxPool%functional_1/re_lu/Relu:activations:0*/
_output_shapes
:?????????0`*
ksize
*
paddingVALID*
strides
2$
"functional_1/max_pooling2d/MaxPoolΡ
)functional_1/conv_2/Conv2D/ReadVariableOpReadVariableOp2functional_1_conv_2_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02+
)functional_1/conv_2/Conv2D/ReadVariableOp
functional_1/conv_2/Conv2DConv2D+functional_1/max_pooling2d/MaxPool:output:01functional_1/conv_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????0` *
paddingSAME*
strides
2
functional_1/conv_2/Conv2DΘ
*functional_1/conv_2/BiasAdd/ReadVariableOpReadVariableOp3functional_1_conv_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02,
*functional_1/conv_2/BiasAdd/ReadVariableOpΨ
functional_1/conv_2/BiasAddBiasAdd#functional_1/conv_2/Conv2D:output:02functional_1/conv_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????0` 2
functional_1/conv_2/BiasAddέ
1functional_1/batch_normalization_1/ReadVariableOpReadVariableOp:functional_1_batch_normalization_1_readvariableop_resource*
_output_shapes
: *
dtype023
1functional_1/batch_normalization_1/ReadVariableOpγ
3functional_1/batch_normalization_1/ReadVariableOp_1ReadVariableOp<functional_1_batch_normalization_1_readvariableop_1_resource*
_output_shapes
: *
dtype025
3functional_1/batch_normalization_1/ReadVariableOp_1
Bfunctional_1/batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOpKfunctional_1_batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02D
Bfunctional_1/batch_normalization_1/FusedBatchNormV3/ReadVariableOp
Dfunctional_1/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpMfunctional_1_batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02F
Dfunctional_1/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1Ί
3functional_1/batch_normalization_1/FusedBatchNormV3FusedBatchNormV3$functional_1/conv_2/BiasAdd:output:09functional_1/batch_normalization_1/ReadVariableOp:value:0;functional_1/batch_normalization_1/ReadVariableOp_1:value:0Jfunctional_1/batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0Lfunctional_1/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????0` : : : : :*
epsilon%o:*
is_training( 25
3functional_1/batch_normalization_1/FusedBatchNormV3±
functional_1/re_lu_1/ReluRelu7functional_1/batch_normalization_1/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????0` 2
functional_1/re_lu_1/Reluν
$functional_1/max_pooling2d_1/MaxPoolMaxPool'functional_1/re_lu_1/Relu:activations:0*/
_output_shapes
:?????????0 *
ksize
*
paddingVALID*
strides
2&
$functional_1/max_pooling2d_1/MaxPoolΡ
)functional_1/conv_3/Conv2D/ReadVariableOpReadVariableOp2functional_1_conv_3_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02+
)functional_1/conv_3/Conv2D/ReadVariableOp
functional_1/conv_3/Conv2DConv2D-functional_1/max_pooling2d_1/MaxPool:output:01functional_1/conv_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????0@*
paddingSAME*
strides
2
functional_1/conv_3/Conv2DΘ
*functional_1/conv_3/BiasAdd/ReadVariableOpReadVariableOp3functional_1_conv_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02,
*functional_1/conv_3/BiasAdd/ReadVariableOpΨ
functional_1/conv_3/BiasAddBiasAdd#functional_1/conv_3/Conv2D:output:02functional_1/conv_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????0@2
functional_1/conv_3/BiasAddέ
1functional_1/batch_normalization_2/ReadVariableOpReadVariableOp:functional_1_batch_normalization_2_readvariableop_resource*
_output_shapes
:@*
dtype023
1functional_1/batch_normalization_2/ReadVariableOpγ
3functional_1/batch_normalization_2/ReadVariableOp_1ReadVariableOp<functional_1_batch_normalization_2_readvariableop_1_resource*
_output_shapes
:@*
dtype025
3functional_1/batch_normalization_2/ReadVariableOp_1
Bfunctional_1/batch_normalization_2/FusedBatchNormV3/ReadVariableOpReadVariableOpKfunctional_1_batch_normalization_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02D
Bfunctional_1/batch_normalization_2/FusedBatchNormV3/ReadVariableOp
Dfunctional_1/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpMfunctional_1_batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02F
Dfunctional_1/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1Ί
3functional_1/batch_normalization_2/FusedBatchNormV3FusedBatchNormV3$functional_1/conv_3/BiasAdd:output:09functional_1/batch_normalization_2/ReadVariableOp:value:0;functional_1/batch_normalization_2/ReadVariableOp_1:value:0Jfunctional_1/batch_normalization_2/FusedBatchNormV3/ReadVariableOp:value:0Lfunctional_1/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????0@:@:@:@:@:*
epsilon%o:*
is_training( 25
3functional_1/batch_normalization_2/FusedBatchNormV3±
functional_1/re_lu_2/ReluRelu7functional_1/batch_normalization_2/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????0@2
functional_1/re_lu_2/Reluν
$functional_1/max_pooling2d_2/MaxPoolMaxPool'functional_1/re_lu_2/Relu:activations:0*/
_output_shapes
:?????????@*
ksize
*
paddingVALID*
strides
2&
$functional_1/max_pooling2d_2/MaxPool?
)functional_1/conv_4/Conv2D/ReadVariableOpReadVariableOp2functional_1_conv_4_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype02+
)functional_1/conv_4/Conv2D/ReadVariableOp
functional_1/conv_4/Conv2DConv2D-functional_1/max_pooling2d_2/MaxPool:output:01functional_1/conv_4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????*
paddingSAME*
strides
2
functional_1/conv_4/Conv2DΙ
*functional_1/conv_4/BiasAdd/ReadVariableOpReadVariableOp3functional_1_conv_4_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02,
*functional_1/conv_4/BiasAdd/ReadVariableOpΩ
functional_1/conv_4/BiasAddBiasAdd#functional_1/conv_4/Conv2D:output:02functional_1/conv_4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????2
functional_1/conv_4/BiasAddή
1functional_1/batch_normalization_3/ReadVariableOpReadVariableOp:functional_1_batch_normalization_3_readvariableop_resource*
_output_shapes	
:*
dtype023
1functional_1/batch_normalization_3/ReadVariableOpδ
3functional_1/batch_normalization_3/ReadVariableOp_1ReadVariableOp<functional_1_batch_normalization_3_readvariableop_1_resource*
_output_shapes	
:*
dtype025
3functional_1/batch_normalization_3/ReadVariableOp_1
Bfunctional_1/batch_normalization_3/FusedBatchNormV3/ReadVariableOpReadVariableOpKfunctional_1_batch_normalization_3_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02D
Bfunctional_1/batch_normalization_3/FusedBatchNormV3/ReadVariableOp
Dfunctional_1/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpMfunctional_1_batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02F
Dfunctional_1/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1Ώ
3functional_1/batch_normalization_3/FusedBatchNormV3FusedBatchNormV3$functional_1/conv_4/BiasAdd:output:09functional_1/batch_normalization_3/ReadVariableOp:value:0;functional_1/batch_normalization_3/ReadVariableOp_1:value:0Jfunctional_1/batch_normalization_3/FusedBatchNormV3/ReadVariableOp:value:0Lfunctional_1/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:?????????:::::*
epsilon%o:*
is_training( 25
3functional_1/batch_normalization_3/FusedBatchNormV3²
functional_1/re_lu_3/ReluRelu7functional_1/batch_normalization_3/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:?????????2
functional_1/re_lu_3/Reluξ
$functional_1/max_pooling2d_3/MaxPoolMaxPool'functional_1/re_lu_3/Relu:activations:0*0
_output_shapes
:?????????*
ksize
*
paddingVALID*
strides
2&
$functional_1/max_pooling2d_3/MaxPoolΣ
)functional_1/conv_5/Conv2D/ReadVariableOpReadVariableOp2functional_1_conv_5_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02+
)functional_1/conv_5/Conv2D/ReadVariableOp
functional_1/conv_5/Conv2DConv2D-functional_1/max_pooling2d_3/MaxPool:output:01functional_1/conv_5/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????*
paddingSAME*
strides
2
functional_1/conv_5/Conv2DΙ
*functional_1/conv_5/BiasAdd/ReadVariableOpReadVariableOp3functional_1_conv_5_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02,
*functional_1/conv_5/BiasAdd/ReadVariableOpΩ
functional_1/conv_5/BiasAddBiasAdd#functional_1/conv_5/Conv2D:output:02functional_1/conv_5/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????2
functional_1/conv_5/BiasAddή
1functional_1/batch_normalization_4/ReadVariableOpReadVariableOp:functional_1_batch_normalization_4_readvariableop_resource*
_output_shapes	
:*
dtype023
1functional_1/batch_normalization_4/ReadVariableOpδ
3functional_1/batch_normalization_4/ReadVariableOp_1ReadVariableOp<functional_1_batch_normalization_4_readvariableop_1_resource*
_output_shapes	
:*
dtype025
3functional_1/batch_normalization_4/ReadVariableOp_1
Bfunctional_1/batch_normalization_4/FusedBatchNormV3/ReadVariableOpReadVariableOpKfunctional_1_batch_normalization_4_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02D
Bfunctional_1/batch_normalization_4/FusedBatchNormV3/ReadVariableOp
Dfunctional_1/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpMfunctional_1_batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02F
Dfunctional_1/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1Ώ
3functional_1/batch_normalization_4/FusedBatchNormV3FusedBatchNormV3$functional_1/conv_5/BiasAdd:output:09functional_1/batch_normalization_4/ReadVariableOp:value:0;functional_1/batch_normalization_4/ReadVariableOp_1:value:0Jfunctional_1/batch_normalization_4/FusedBatchNormV3/ReadVariableOp:value:0Lfunctional_1/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:?????????:::::*
epsilon%o:*
is_training( 25
3functional_1/batch_normalization_4/FusedBatchNormV3²
functional_1/re_lu_4/ReluRelu7functional_1/batch_normalization_4/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:?????????2
functional_1/re_lu_4/Relu
functional_1/upsample_6/ShapeShape'functional_1/re_lu_4/Relu:activations:0*
T0*
_output_shapes
:2
functional_1/upsample_6/Shape€
+functional_1/upsample_6/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2-
+functional_1/upsample_6/strided_slice/stack¨
-functional_1/upsample_6/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2/
-functional_1/upsample_6/strided_slice/stack_1¨
-functional_1/upsample_6/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-functional_1/upsample_6/strided_slice/stack_2ς
%functional_1/upsample_6/strided_sliceStridedSlice&functional_1/upsample_6/Shape:output:04functional_1/upsample_6/strided_slice/stack:output:06functional_1/upsample_6/strided_slice/stack_1:output:06functional_1/upsample_6/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2'
%functional_1/upsample_6/strided_slice
functional_1/upsample_6/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2!
functional_1/upsample_6/stack/1
functional_1/upsample_6/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2!
functional_1/upsample_6/stack/2
functional_1/upsample_6/stack/3Const*
_output_shapes
: *
dtype0*
value
B :2!
functional_1/upsample_6/stack/3’
functional_1/upsample_6/stackPack.functional_1/upsample_6/strided_slice:output:0(functional_1/upsample_6/stack/1:output:0(functional_1/upsample_6/stack/2:output:0(functional_1/upsample_6/stack/3:output:0*
N*
T0*
_output_shapes
:2
functional_1/upsample_6/stack¨
-functional_1/upsample_6/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2/
-functional_1/upsample_6/strided_slice_1/stack¬
/functional_1/upsample_6/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:21
/functional_1/upsample_6/strided_slice_1/stack_1¬
/functional_1/upsample_6/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:21
/functional_1/upsample_6/strided_slice_1/stack_2ό
'functional_1/upsample_6/strided_slice_1StridedSlice&functional_1/upsample_6/stack:output:06functional_1/upsample_6/strided_slice_1/stack:output:08functional_1/upsample_6/strided_slice_1/stack_1:output:08functional_1/upsample_6/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2)
'functional_1/upsample_6/strided_slice_1ύ
7functional_1/upsample_6/conv2d_transpose/ReadVariableOpReadVariableOp@functional_1_upsample_6_conv2d_transpose_readvariableop_resource*(
_output_shapes
:*
dtype029
7functional_1/upsample_6/conv2d_transpose/ReadVariableOpα
(functional_1/upsample_6/conv2d_transposeConv2DBackpropInput&functional_1/upsample_6/stack:output:0?functional_1/upsample_6/conv2d_transpose/ReadVariableOp:value:0'functional_1/re_lu_4/Relu:activations:0*
T0*0
_output_shapes
:?????????*
paddingVALID*
strides
2*
(functional_1/upsample_6/conv2d_transposeΥ
.functional_1/upsample_6/BiasAdd/ReadVariableOpReadVariableOp7functional_1_upsample_6_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype020
.functional_1/upsample_6/BiasAdd/ReadVariableOpσ
functional_1/upsample_6/BiasAddBiasAdd1functional_1/upsample_6/conv2d_transpose:output:06functional_1/upsample_6/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????2!
functional_1/upsample_6/BiasAdd­
/functional_1/tf_op_layer_concat_6/concat_6/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????21
/functional_1/tf_op_layer_concat_6/concat_6/axisΔ
*functional_1/tf_op_layer_concat_6/concat_6ConcatV2(functional_1/upsample_6/BiasAdd:output:0'functional_1/re_lu_3/Relu:activations:08functional_1/tf_op_layer_concat_6/concat_6/axis:output:0*
N*
T0*
_cloned(*0
_output_shapes
:?????????2,
*functional_1/tf_op_layer_concat_6/concat_6Σ
)functional_1/conv_6/Conv2D/ReadVariableOpReadVariableOp2functional_1_conv_6_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02+
)functional_1/conv_6/Conv2D/ReadVariableOp
functional_1/conv_6/Conv2DConv2D3functional_1/tf_op_layer_concat_6/concat_6:output:01functional_1/conv_6/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????*
paddingSAME*
strides
2
functional_1/conv_6/Conv2DΙ
*functional_1/conv_6/BiasAdd/ReadVariableOpReadVariableOp3functional_1_conv_6_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02,
*functional_1/conv_6/BiasAdd/ReadVariableOpΩ
functional_1/conv_6/BiasAddBiasAdd#functional_1/conv_6/Conv2D:output:02functional_1/conv_6/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????2
functional_1/conv_6/BiasAddή
1functional_1/batch_normalization_5/ReadVariableOpReadVariableOp:functional_1_batch_normalization_5_readvariableop_resource*
_output_shapes	
:*
dtype023
1functional_1/batch_normalization_5/ReadVariableOpδ
3functional_1/batch_normalization_5/ReadVariableOp_1ReadVariableOp<functional_1_batch_normalization_5_readvariableop_1_resource*
_output_shapes	
:*
dtype025
3functional_1/batch_normalization_5/ReadVariableOp_1
Bfunctional_1/batch_normalization_5/FusedBatchNormV3/ReadVariableOpReadVariableOpKfunctional_1_batch_normalization_5_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02D
Bfunctional_1/batch_normalization_5/FusedBatchNormV3/ReadVariableOp
Dfunctional_1/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpMfunctional_1_batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02F
Dfunctional_1/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1Ώ
3functional_1/batch_normalization_5/FusedBatchNormV3FusedBatchNormV3$functional_1/conv_6/BiasAdd:output:09functional_1/batch_normalization_5/ReadVariableOp:value:0;functional_1/batch_normalization_5/ReadVariableOp_1:value:0Jfunctional_1/batch_normalization_5/FusedBatchNormV3/ReadVariableOp:value:0Lfunctional_1/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:?????????:::::*
epsilon%o:*
is_training( 25
3functional_1/batch_normalization_5/FusedBatchNormV3²
functional_1/re_lu_5/ReluRelu7functional_1/batch_normalization_5/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:?????????2
functional_1/re_lu_5/Relu
functional_1/upsample_7/ShapeShape'functional_1/re_lu_5/Relu:activations:0*
T0*
_output_shapes
:2
functional_1/upsample_7/Shape€
+functional_1/upsample_7/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2-
+functional_1/upsample_7/strided_slice/stack¨
-functional_1/upsample_7/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2/
-functional_1/upsample_7/strided_slice/stack_1¨
-functional_1/upsample_7/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-functional_1/upsample_7/strided_slice/stack_2ς
%functional_1/upsample_7/strided_sliceStridedSlice&functional_1/upsample_7/Shape:output:04functional_1/upsample_7/strided_slice/stack:output:06functional_1/upsample_7/strided_slice/stack_1:output:06functional_1/upsample_7/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2'
%functional_1/upsample_7/strided_slice
functional_1/upsample_7/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2!
functional_1/upsample_7/stack/1
functional_1/upsample_7/stack/2Const*
_output_shapes
: *
dtype0*
value	B :02!
functional_1/upsample_7/stack/2
functional_1/upsample_7/stack/3Const*
_output_shapes
: *
dtype0*
value	B :@2!
functional_1/upsample_7/stack/3’
functional_1/upsample_7/stackPack.functional_1/upsample_7/strided_slice:output:0(functional_1/upsample_7/stack/1:output:0(functional_1/upsample_7/stack/2:output:0(functional_1/upsample_7/stack/3:output:0*
N*
T0*
_output_shapes
:2
functional_1/upsample_7/stack¨
-functional_1/upsample_7/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2/
-functional_1/upsample_7/strided_slice_1/stack¬
/functional_1/upsample_7/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:21
/functional_1/upsample_7/strided_slice_1/stack_1¬
/functional_1/upsample_7/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:21
/functional_1/upsample_7/strided_slice_1/stack_2ό
'functional_1/upsample_7/strided_slice_1StridedSlice&functional_1/upsample_7/stack:output:06functional_1/upsample_7/strided_slice_1/stack:output:08functional_1/upsample_7/strided_slice_1/stack_1:output:08functional_1/upsample_7/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2)
'functional_1/upsample_7/strided_slice_1ό
7functional_1/upsample_7/conv2d_transpose/ReadVariableOpReadVariableOp@functional_1_upsample_7_conv2d_transpose_readvariableop_resource*'
_output_shapes
:@*
dtype029
7functional_1/upsample_7/conv2d_transpose/ReadVariableOpΰ
(functional_1/upsample_7/conv2d_transposeConv2DBackpropInput&functional_1/upsample_7/stack:output:0?functional_1/upsample_7/conv2d_transpose/ReadVariableOp:value:0'functional_1/re_lu_5/Relu:activations:0*
T0*/
_output_shapes
:?????????0@*
paddingVALID*
strides
2*
(functional_1/upsample_7/conv2d_transposeΤ
.functional_1/upsample_7/BiasAdd/ReadVariableOpReadVariableOp7functional_1_upsample_7_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype020
.functional_1/upsample_7/BiasAdd/ReadVariableOpς
functional_1/upsample_7/BiasAddBiasAdd1functional_1/upsample_7/conv2d_transpose:output:06functional_1/upsample_7/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????0@2!
functional_1/upsample_7/BiasAdd­
/functional_1/tf_op_layer_concat_7/concat_7/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????21
/functional_1/tf_op_layer_concat_7/concat_7/axisΔ
*functional_1/tf_op_layer_concat_7/concat_7ConcatV2(functional_1/upsample_7/BiasAdd:output:0'functional_1/re_lu_2/Relu:activations:08functional_1/tf_op_layer_concat_7/concat_7/axis:output:0*
N*
T0*
_cloned(*0
_output_shapes
:?????????02,
*functional_1/tf_op_layer_concat_7/concat_7?
)functional_1/conv_7/Conv2D/ReadVariableOpReadVariableOp2functional_1_conv_7_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype02+
)functional_1/conv_7/Conv2D/ReadVariableOp
functional_1/conv_7/Conv2DConv2D3functional_1/tf_op_layer_concat_7/concat_7:output:01functional_1/conv_7/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????0@*
paddingSAME*
strides
2
functional_1/conv_7/Conv2DΘ
*functional_1/conv_7/BiasAdd/ReadVariableOpReadVariableOp3functional_1_conv_7_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02,
*functional_1/conv_7/BiasAdd/ReadVariableOpΨ
functional_1/conv_7/BiasAddBiasAdd#functional_1/conv_7/Conv2D:output:02functional_1/conv_7/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????0@2
functional_1/conv_7/BiasAddέ
1functional_1/batch_normalization_6/ReadVariableOpReadVariableOp:functional_1_batch_normalization_6_readvariableop_resource*
_output_shapes
:@*
dtype023
1functional_1/batch_normalization_6/ReadVariableOpγ
3functional_1/batch_normalization_6/ReadVariableOp_1ReadVariableOp<functional_1_batch_normalization_6_readvariableop_1_resource*
_output_shapes
:@*
dtype025
3functional_1/batch_normalization_6/ReadVariableOp_1
Bfunctional_1/batch_normalization_6/FusedBatchNormV3/ReadVariableOpReadVariableOpKfunctional_1_batch_normalization_6_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02D
Bfunctional_1/batch_normalization_6/FusedBatchNormV3/ReadVariableOp
Dfunctional_1/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpMfunctional_1_batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02F
Dfunctional_1/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1Ί
3functional_1/batch_normalization_6/FusedBatchNormV3FusedBatchNormV3$functional_1/conv_7/BiasAdd:output:09functional_1/batch_normalization_6/ReadVariableOp:value:0;functional_1/batch_normalization_6/ReadVariableOp_1:value:0Jfunctional_1/batch_normalization_6/FusedBatchNormV3/ReadVariableOp:value:0Lfunctional_1/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????0@:@:@:@:@:*
epsilon%o:*
is_training( 25
3functional_1/batch_normalization_6/FusedBatchNormV3±
functional_1/re_lu_6/ReluRelu7functional_1/batch_normalization_6/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????0@2
functional_1/re_lu_6/Relu
functional_1/upsample_8/ShapeShape'functional_1/re_lu_6/Relu:activations:0*
T0*
_output_shapes
:2
functional_1/upsample_8/Shape€
+functional_1/upsample_8/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2-
+functional_1/upsample_8/strided_slice/stack¨
-functional_1/upsample_8/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2/
-functional_1/upsample_8/strided_slice/stack_1¨
-functional_1/upsample_8/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-functional_1/upsample_8/strided_slice/stack_2ς
%functional_1/upsample_8/strided_sliceStridedSlice&functional_1/upsample_8/Shape:output:04functional_1/upsample_8/strided_slice/stack:output:06functional_1/upsample_8/strided_slice/stack_1:output:06functional_1/upsample_8/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2'
%functional_1/upsample_8/strided_slice
functional_1/upsample_8/stack/1Const*
_output_shapes
: *
dtype0*
value	B :02!
functional_1/upsample_8/stack/1
functional_1/upsample_8/stack/2Const*
_output_shapes
: *
dtype0*
value	B :`2!
functional_1/upsample_8/stack/2
functional_1/upsample_8/stack/3Const*
_output_shapes
: *
dtype0*
value	B : 2!
functional_1/upsample_8/stack/3’
functional_1/upsample_8/stackPack.functional_1/upsample_8/strided_slice:output:0(functional_1/upsample_8/stack/1:output:0(functional_1/upsample_8/stack/2:output:0(functional_1/upsample_8/stack/3:output:0*
N*
T0*
_output_shapes
:2
functional_1/upsample_8/stack¨
-functional_1/upsample_8/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2/
-functional_1/upsample_8/strided_slice_1/stack¬
/functional_1/upsample_8/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:21
/functional_1/upsample_8/strided_slice_1/stack_1¬
/functional_1/upsample_8/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:21
/functional_1/upsample_8/strided_slice_1/stack_2ό
'functional_1/upsample_8/strided_slice_1StridedSlice&functional_1/upsample_8/stack:output:06functional_1/upsample_8/strided_slice_1/stack:output:08functional_1/upsample_8/strided_slice_1/stack_1:output:08functional_1/upsample_8/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2)
'functional_1/upsample_8/strided_slice_1ϋ
7functional_1/upsample_8/conv2d_transpose/ReadVariableOpReadVariableOp@functional_1_upsample_8_conv2d_transpose_readvariableop_resource*&
_output_shapes
: @*
dtype029
7functional_1/upsample_8/conv2d_transpose/ReadVariableOpΰ
(functional_1/upsample_8/conv2d_transposeConv2DBackpropInput&functional_1/upsample_8/stack:output:0?functional_1/upsample_8/conv2d_transpose/ReadVariableOp:value:0'functional_1/re_lu_6/Relu:activations:0*
T0*/
_output_shapes
:?????????0` *
paddingVALID*
strides
2*
(functional_1/upsample_8/conv2d_transposeΤ
.functional_1/upsample_8/BiasAdd/ReadVariableOpReadVariableOp7functional_1_upsample_8_biasadd_readvariableop_resource*
_output_shapes
: *
dtype020
.functional_1/upsample_8/BiasAdd/ReadVariableOpς
functional_1/upsample_8/BiasAddBiasAdd1functional_1/upsample_8/conv2d_transpose:output:06functional_1/upsample_8/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????0` 2!
functional_1/upsample_8/BiasAdd­
/functional_1/tf_op_layer_concat_8/concat_8/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????21
/functional_1/tf_op_layer_concat_8/concat_8/axisΓ
*functional_1/tf_op_layer_concat_8/concat_8ConcatV2(functional_1/upsample_8/BiasAdd:output:0'functional_1/re_lu_1/Relu:activations:08functional_1/tf_op_layer_concat_8/concat_8/axis:output:0*
N*
T0*
_cloned(*/
_output_shapes
:?????????0`@2,
*functional_1/tf_op_layer_concat_8/concat_8Ρ
)functional_1/conv_8/Conv2D/ReadVariableOpReadVariableOp2functional_1_conv_8_conv2d_readvariableop_resource*&
_output_shapes
:@ *
dtype02+
)functional_1/conv_8/Conv2D/ReadVariableOp
functional_1/conv_8/Conv2DConv2D3functional_1/tf_op_layer_concat_8/concat_8:output:01functional_1/conv_8/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????0` *
paddingSAME*
strides
2
functional_1/conv_8/Conv2DΘ
*functional_1/conv_8/BiasAdd/ReadVariableOpReadVariableOp3functional_1_conv_8_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02,
*functional_1/conv_8/BiasAdd/ReadVariableOpΨ
functional_1/conv_8/BiasAddBiasAdd#functional_1/conv_8/Conv2D:output:02functional_1/conv_8/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????0` 2
functional_1/conv_8/BiasAddέ
1functional_1/batch_normalization_7/ReadVariableOpReadVariableOp:functional_1_batch_normalization_7_readvariableop_resource*
_output_shapes
: *
dtype023
1functional_1/batch_normalization_7/ReadVariableOpγ
3functional_1/batch_normalization_7/ReadVariableOp_1ReadVariableOp<functional_1_batch_normalization_7_readvariableop_1_resource*
_output_shapes
: *
dtype025
3functional_1/batch_normalization_7/ReadVariableOp_1
Bfunctional_1/batch_normalization_7/FusedBatchNormV3/ReadVariableOpReadVariableOpKfunctional_1_batch_normalization_7_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02D
Bfunctional_1/batch_normalization_7/FusedBatchNormV3/ReadVariableOp
Dfunctional_1/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpMfunctional_1_batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02F
Dfunctional_1/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1Ί
3functional_1/batch_normalization_7/FusedBatchNormV3FusedBatchNormV3$functional_1/conv_8/BiasAdd:output:09functional_1/batch_normalization_7/ReadVariableOp:value:0;functional_1/batch_normalization_7/ReadVariableOp_1:value:0Jfunctional_1/batch_normalization_7/FusedBatchNormV3/ReadVariableOp:value:0Lfunctional_1/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????0` : : : : :*
epsilon%o:*
is_training( 25
3functional_1/batch_normalization_7/FusedBatchNormV3±
functional_1/re_lu_7/ReluRelu7functional_1/batch_normalization_7/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????0` 2
functional_1/re_lu_7/Relu
functional_1/upsample_9/ShapeShape'functional_1/re_lu_7/Relu:activations:0*
T0*
_output_shapes
:2
functional_1/upsample_9/Shape€
+functional_1/upsample_9/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2-
+functional_1/upsample_9/strided_slice/stack¨
-functional_1/upsample_9/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2/
-functional_1/upsample_9/strided_slice/stack_1¨
-functional_1/upsample_9/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-functional_1/upsample_9/strided_slice/stack_2ς
%functional_1/upsample_9/strided_sliceStridedSlice&functional_1/upsample_9/Shape:output:04functional_1/upsample_9/strided_slice/stack:output:06functional_1/upsample_9/strided_slice/stack_1:output:06functional_1/upsample_9/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2'
%functional_1/upsample_9/strided_slice
functional_1/upsample_9/stack/1Const*
_output_shapes
: *
dtype0*
value	B :`2!
functional_1/upsample_9/stack/1
functional_1/upsample_9/stack/2Const*
_output_shapes
: *
dtype0*
value
B :ΐ2!
functional_1/upsample_9/stack/2
functional_1/upsample_9/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2!
functional_1/upsample_9/stack/3’
functional_1/upsample_9/stackPack.functional_1/upsample_9/strided_slice:output:0(functional_1/upsample_9/stack/1:output:0(functional_1/upsample_9/stack/2:output:0(functional_1/upsample_9/stack/3:output:0*
N*
T0*
_output_shapes
:2
functional_1/upsample_9/stack¨
-functional_1/upsample_9/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2/
-functional_1/upsample_9/strided_slice_1/stack¬
/functional_1/upsample_9/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:21
/functional_1/upsample_9/strided_slice_1/stack_1¬
/functional_1/upsample_9/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:21
/functional_1/upsample_9/strided_slice_1/stack_2ό
'functional_1/upsample_9/strided_slice_1StridedSlice&functional_1/upsample_9/stack:output:06functional_1/upsample_9/strided_slice_1/stack:output:08functional_1/upsample_9/strided_slice_1/stack_1:output:08functional_1/upsample_9/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2)
'functional_1/upsample_9/strided_slice_1ϋ
7functional_1/upsample_9/conv2d_transpose/ReadVariableOpReadVariableOp@functional_1_upsample_9_conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype029
7functional_1/upsample_9/conv2d_transpose/ReadVariableOpα
(functional_1/upsample_9/conv2d_transposeConv2DBackpropInput&functional_1/upsample_9/stack:output:0?functional_1/upsample_9/conv2d_transpose/ReadVariableOp:value:0'functional_1/re_lu_7/Relu:activations:0*
T0*0
_output_shapes
:?????????`ΐ*
paddingVALID*
strides
2*
(functional_1/upsample_9/conv2d_transposeΤ
.functional_1/upsample_9/BiasAdd/ReadVariableOpReadVariableOp7functional_1_upsample_9_biasadd_readvariableop_resource*
_output_shapes
:*
dtype020
.functional_1/upsample_9/BiasAdd/ReadVariableOpσ
functional_1/upsample_9/BiasAddBiasAdd1functional_1/upsample_9/conv2d_transpose:output:06functional_1/upsample_9/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????`ΐ2!
functional_1/upsample_9/BiasAdd­
/functional_1/tf_op_layer_concat_9/concat_9/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????21
/functional_1/tf_op_layer_concat_9/concat_9/axisΒ
*functional_1/tf_op_layer_concat_9/concat_9ConcatV2(functional_1/upsample_9/BiasAdd:output:0%functional_1/re_lu/Relu:activations:08functional_1/tf_op_layer_concat_9/concat_9/axis:output:0*
N*
T0*
_cloned(*0
_output_shapes
:?????????`ΐ 2,
*functional_1/tf_op_layer_concat_9/concat_9Ρ
)functional_1/conv_9/Conv2D/ReadVariableOpReadVariableOp2functional_1_conv_9_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02+
)functional_1/conv_9/Conv2D/ReadVariableOp
functional_1/conv_9/Conv2DConv2D3functional_1/tf_op_layer_concat_9/concat_9:output:01functional_1/conv_9/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????`ΐ*
paddingSAME*
strides
2
functional_1/conv_9/Conv2DΘ
*functional_1/conv_9/BiasAdd/ReadVariableOpReadVariableOp3functional_1_conv_9_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02,
*functional_1/conv_9/BiasAdd/ReadVariableOpΩ
functional_1/conv_9/BiasAddBiasAdd#functional_1/conv_9/Conv2D:output:02functional_1/conv_9/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????`ΐ2
functional_1/conv_9/BiasAddέ
1functional_1/batch_normalization_8/ReadVariableOpReadVariableOp:functional_1_batch_normalization_8_readvariableop_resource*
_output_shapes
:*
dtype023
1functional_1/batch_normalization_8/ReadVariableOpγ
3functional_1/batch_normalization_8/ReadVariableOp_1ReadVariableOp<functional_1_batch_normalization_8_readvariableop_1_resource*
_output_shapes
:*
dtype025
3functional_1/batch_normalization_8/ReadVariableOp_1
Bfunctional_1/batch_normalization_8/FusedBatchNormV3/ReadVariableOpReadVariableOpKfunctional_1_batch_normalization_8_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02D
Bfunctional_1/batch_normalization_8/FusedBatchNormV3/ReadVariableOp
Dfunctional_1/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpMfunctional_1_batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02F
Dfunctional_1/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1»
3functional_1/batch_normalization_8/FusedBatchNormV3FusedBatchNormV3$functional_1/conv_9/BiasAdd:output:09functional_1/batch_normalization_8/ReadVariableOp:value:0;functional_1/batch_normalization_8/ReadVariableOp_1:value:0Jfunctional_1/batch_normalization_8/FusedBatchNormV3/ReadVariableOp:value:0Lfunctional_1/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*L
_output_shapes:
8:?????????`ΐ:::::*
epsilon%o:*
is_training( 25
3functional_1/batch_normalization_8/FusedBatchNormV3²
functional_1/re_lu_8/ReluRelu7functional_1/batch_normalization_8/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:?????????`ΐ2
functional_1/re_lu_8/ReluΞ
(functional_1/final/Conv2D/ReadVariableOpReadVariableOp1functional_1_final_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02*
(functional_1/final/Conv2D/ReadVariableOpώ
functional_1/final/Conv2DConv2D'functional_1/re_lu_8/Relu:activations:00functional_1/final/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????`ΐ*
paddingSAME*
strides
2
functional_1/final/Conv2DΕ
)functional_1/final/BiasAdd/ReadVariableOpReadVariableOp2functional_1_final_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02+
)functional_1/final/BiasAdd/ReadVariableOpΥ
functional_1/final/BiasAddBiasAdd"functional_1/final/Conv2D:output:01functional_1/final/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????`ΐ2
functional_1/final/BiasAdd
IdentityIdentity#functional_1/final/BiasAdd:output:0*
T0*0
_output_shapes
:?????????`ΐ2

Identity"
identityIdentity:output:0*±
_input_shapes
:?????????`ΐ:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::\ X
0
_output_shapes
:?????????`ΐ
$
_user_specified_name
imageInput
ή
©
6__inference_batch_normalization_layer_call_fn_57381356

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity’StatefulPartitionedCall§
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????`ΐ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *Z
fURS
Q__inference_batch_normalization_layer_call_and_return_conditional_losses_573783322
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:?????????`ΐ2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:?????????`ΐ::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:?????????`ΐ
 
_user_specified_nameinputs

?
Q__inference_batch_normalization_layer_call_and_return_conditional_losses_57378332

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity’AssignNewValue’AssignNewValue_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ω
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*L
_output_shapes:
8:?????????`ΐ:::::*
epsilon%o:*
exponential_avg_factor%
Χ#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*0
_output_shapes
:?????????`ΐ2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:?????????`ΐ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:X T
0
_output_shapes
:?????????`ΐ
 
_user_specified_nameinputs
β
«
8__inference_batch_normalization_4_layer_call_fn_57381920

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity’StatefulPartitionedCall©
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *\
fWRU
S__inference_batch_normalization_4_layer_call_and_return_conditional_losses_573787842
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:?????????::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:?????????
 
_user_specified_nameinputs
Τ

-__inference_upsample_7_layer_call_fn_57377847

inputs
unknown
	unknown_0
identity’StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *Q
fLRJ
H__inference_upsample_7_layer_call_and_return_conditional_losses_573778372
StatefulPartitionedCall¨
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:,???????????????????????????::22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,???????????????????????????
 
_user_specified_nameinputs
»
|
R__inference_tf_op_layer_concat_9_layer_call_and_return_conditional_losses_57379262

inputs
inputs_1
identityi
concat_9/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
concat_9/axis
concat_9ConcatV2inputsinputs_1concat_9/axis:output:0*
N*
T0*
_cloned(*0
_output_shapes
:?????????`ΐ 2

concat_9n
IdentityIdentityconcat_9:output:0*
T0*0
_output_shapes
:?????????`ΐ 2

Identity"
identityIdentity:output:0*\
_input_shapesK
I:+???????????????????????????:?????????`ΐ:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs:XT
0
_output_shapes
:?????????`ΐ
 
_user_specified_nameinputs

°
S__inference_batch_normalization_3_layer_call_and_return_conditional_losses_57381796

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity’AssignNewValue’AssignNewValue_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype02
ReadVariableOp_1¨
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1έ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:?????????:::::*
epsilon%o:*
exponential_avg_factor%
Χ#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*0
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:?????????::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:X T
0
_output_shapes
:?????????
 
_user_specified_nameinputs
ζ
m
Q__inference_tf_op_layer_RealDiv_layer_call_and_return_conditional_losses_57378265

inputs
identity[
	RealDiv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?B2
	RealDiv/y
RealDivRealDivinputsRealDiv/y:output:0*
T0*
_cloned(*0
_output_shapes
:?????????`ΐ2	
RealDivh
IdentityIdentityRealDiv:z:0*
T0*0
_output_shapes
:?????????`ΐ2

Identity"
identityIdentity:output:0*/
_input_shapes
:?????????`ΐ:X T
0
_output_shapes
:?????????`ΐ
 
_user_specified_nameinputs
§

S__inference_batch_normalization_5_layer_call_and_return_conditional_losses_57382141

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityu
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype02
ReadVariableOp_1¨
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1α
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,???????????????????????????:::::*
epsilon%o:*
is_training( 2
FusedBatchNormV3
IdentityIdentityFusedBatchNormV3:y:0*
T0*B
_output_shapes0
.:,???????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,???????????????????????????:::::j f
B
_output_shapes0
.:,???????????????????????????
 
_user_specified_nameinputs
ΰ
©
6__inference_batch_normalization_layer_call_fn_57381369

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity’StatefulPartitionedCall©
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????`ΐ*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *Z
fURS
Q__inference_batch_normalization_layer_call_and_return_conditional_losses_573783502
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:?????????`ΐ2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:?????????`ΐ::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:?????????`ΐ
 
_user_specified_nameinputs
€
©
6__inference_batch_normalization_layer_call_fn_57381305

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity’StatefulPartitionedCallΊ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *Z
fURS
Q__inference_batch_normalization_layer_call_and_return_conditional_losses_573771722
StatefulPartitionedCall¨
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
Χ
°
S__inference_batch_normalization_4_layer_call_and_return_conditional_losses_57381953

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity’AssignNewValue’AssignNewValue_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype02
ReadVariableOp_1¨
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ο
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,???????????????????????????:::::*
epsilon%o:*
exponential_avg_factor%
Χ#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1§
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*B
_output_shapes0
.:,???????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,???????????????????????????::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:j f
B
_output_shapes0
.:,???????????????????????????
 
_user_specified_nameinputs
§

S__inference_batch_normalization_5_layer_call_and_return_conditional_losses_57377788

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityu
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype02
ReadVariableOp_1¨
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1α
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,???????????????????????????:::::*
epsilon%o:*
is_training( 2
FusedBatchNormV3
IdentityIdentityFusedBatchNormV3:y:0*
T0*B
_output_shapes0
.:,???????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,???????????????????????????:::::j f
B
_output_shapes0
.:,???????????????????????????
 
_user_specified_nameinputs
Ή
ί
/__inference_functional_1_layer_call_fn_57379892

imageinput
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

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28

unknown_29

unknown_30

unknown_31

unknown_32

unknown_33

unknown_34

unknown_35

unknown_36

unknown_37

unknown_38

unknown_39

unknown_40

unknown_41

unknown_42

unknown_43

unknown_44

unknown_45

unknown_46

unknown_47

unknown_48

unknown_49

unknown_50

unknown_51

unknown_52

unknown_53

unknown_54

unknown_55

unknown_56

unknown_57

unknown_58

unknown_59

unknown_60

unknown_61

unknown_62
identity’StatefulPartitionedCallΥ	
StatefulPartitionedCallStatefulPartitionedCall
imageinputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46
unknown_47
unknown_48
unknown_49
unknown_50
unknown_51
unknown_52
unknown_53
unknown_54
unknown_55
unknown_56
unknown_57
unknown_58
unknown_59
unknown_60
unknown_61
unknown_62*L
TinE
C2A*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????`ΐ*P
_read_only_resource_inputs2
0.	
 !"#$'()*+,/01234789:;<?@*2
config_proto" 

CPU

GPU2 *0J 8 *S
fNRL
J__inference_functional_1_layer_call_and_return_conditional_losses_573797612
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:?????????`ΐ2

Identity"
identityIdentity:output:0*±
_input_shapes
:?????????`ΐ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:\ X
0
_output_shapes
:?????????`ΐ
$
_user_specified_name
imageInput

c
7__inference_tf_op_layer_concat_6_layer_call_fn_57382020
inputs_0
inputs_1
identityλ
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *[
fVRT
R__inference_tf_op_layer_concat_6_layer_call_and_return_conditional_losses_573788632
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*]
_input_shapesL
J:,???????????????????????????:?????????:l h
B
_output_shapes0
.:,???????????????????????????
"
_user_specified_name
inputs/0:ZV
0
_output_shapes
:?????????
"
_user_specified_name
inputs/1
Φ

S__inference_batch_normalization_8_layer_call_and_return_conditional_losses_57379334

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Λ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*L
_output_shapes:
8:?????????`ΐ:::::*
epsilon%o:*
is_training( 2
FusedBatchNormV3q
IdentityIdentityFusedBatchNormV3:y:0*
T0*0
_output_shapes
:?????????`ΐ2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:?????????`ΐ:::::X T
0
_output_shapes
:?????????`ΐ
 
_user_specified_nameinputs
Λ
°
S__inference_batch_normalization_8_layer_call_and_return_conditional_losses_57382569

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity’AssignNewValue’AssignNewValue_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1κ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o:*
exponential_avg_factor%
Χ#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1¦
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
ή

S__inference_batch_normalization_5_layer_call_and_return_conditional_losses_57382077

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityu
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype02
ReadVariableOp_1¨
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ο
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:?????????:::::*
epsilon%o:*
is_training( 2
FusedBatchNormV3q
IdentityIdentityFusedBatchNormV3:y:0*
T0*0
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:?????????:::::X T
0
_output_shapes
:?????????
 
_user_specified_nameinputs
²
L
0__inference_max_pooling2d_layer_call_fn_57377195

inputs
identityρ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *T
fORM
K__inference_max_pooling2d_layer_call_and_return_conditional_losses_573771892
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
Ά
N
2__inference_max_pooling2d_3_layer_call_fn_57377543

inputs
identityσ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *V
fQRO
M__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_573775372
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?

S__inference_batch_normalization_6_layer_call_and_return_conditional_losses_57379068

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Κ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????0@:@:@:@:@:*
epsilon%o:*
is_training( 2
FusedBatchNormV3p
IdentityIdentityFusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????0@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????0@:::::W S
/
_output_shapes
:?????????0@
 
_user_specified_nameinputs
­
¬
D__inference_conv_6_layer_call_and_return_conditional_losses_57378882

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp€
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????2	
BiasAddm
IdentityIdentityBiasAdd:output:0*
T0*0
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:?????????:::X T
0
_output_shapes
:?????????
 
_user_specified_nameinputs
¬
«
8__inference_batch_normalization_3_layer_call_fn_57381776

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity’StatefulPartitionedCall½
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,???????????????????????????*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *\
fWRU
S__inference_batch_normalization_3_layer_call_and_return_conditional_losses_573775202
StatefulPartitionedCall©
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,???????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,???????????????????????????::::22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,???????????????????????????
 
_user_specified_nameinputs
ή
«
8__inference_batch_normalization_7_layer_call_fn_57382494

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity’StatefulPartitionedCall¨
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????0` *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *\
fWRU
S__inference_batch_normalization_7_layer_call_and_return_conditional_losses_573791832
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????0` 2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????0` ::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????0` 
 
_user_specified_nameinputs
ζ
m
Q__inference_tf_op_layer_RealDiv_layer_call_and_return_conditional_losses_57381206

inputs
identity[
	RealDiv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?B2
	RealDiv/y
RealDivRealDivinputsRealDiv/y:output:0*
T0*
_cloned(*0
_output_shapes
:?????????`ΐ2	
RealDivh
IdentityIdentityRealDiv:z:0*
T0*0
_output_shapes
:?????????`ΐ2

Identity"
identityIdentity:output:0*/
_input_shapes
:?????????`ΐ:X T
0
_output_shapes
:?????????`ΐ
 
_user_specified_nameinputs
Κ
ζ
J__inference_functional_1_layer_call_and_return_conditional_losses_57380068

inputs
conv_1_57379899
conv_1_57379901 
batch_normalization_57379904 
batch_normalization_57379906 
batch_normalization_57379908 
batch_normalization_57379910
conv_2_57379915
conv_2_57379917"
batch_normalization_1_57379920"
batch_normalization_1_57379922"
batch_normalization_1_57379924"
batch_normalization_1_57379926
conv_3_57379931
conv_3_57379933"
batch_normalization_2_57379936"
batch_normalization_2_57379938"
batch_normalization_2_57379940"
batch_normalization_2_57379942
conv_4_57379947
conv_4_57379949"
batch_normalization_3_57379952"
batch_normalization_3_57379954"
batch_normalization_3_57379956"
batch_normalization_3_57379958
conv_5_57379963
conv_5_57379965"
batch_normalization_4_57379968"
batch_normalization_4_57379970"
batch_normalization_4_57379972"
batch_normalization_4_57379974
upsample_6_57379978
upsample_6_57379980
conv_6_57379984
conv_6_57379986"
batch_normalization_5_57379989"
batch_normalization_5_57379991"
batch_normalization_5_57379993"
batch_normalization_5_57379995
upsample_7_57379999
upsample_7_57380001
conv_7_57380005
conv_7_57380007"
batch_normalization_6_57380010"
batch_normalization_6_57380012"
batch_normalization_6_57380014"
batch_normalization_6_57380016
upsample_8_57380020
upsample_8_57380022
conv_8_57380026
conv_8_57380028"
batch_normalization_7_57380031"
batch_normalization_7_57380033"
batch_normalization_7_57380035"
batch_normalization_7_57380037
upsample_9_57380041
upsample_9_57380043
conv_9_57380047
conv_9_57380049"
batch_normalization_8_57380052"
batch_normalization_8_57380054"
batch_normalization_8_57380056"
batch_normalization_8_57380058
final_57380062
final_57380064
identity’+batch_normalization/StatefulPartitionedCall’-batch_normalization_1/StatefulPartitionedCall’-batch_normalization_2/StatefulPartitionedCall’-batch_normalization_3/StatefulPartitionedCall’-batch_normalization_4/StatefulPartitionedCall’-batch_normalization_5/StatefulPartitionedCall’-batch_normalization_6/StatefulPartitionedCall’-batch_normalization_7/StatefulPartitionedCall’-batch_normalization_8/StatefulPartitionedCall’conv_1/StatefulPartitionedCall’conv_2/StatefulPartitionedCall’conv_3/StatefulPartitionedCall’conv_4/StatefulPartitionedCall’conv_5/StatefulPartitionedCall’conv_6/StatefulPartitionedCall’conv_7/StatefulPartitionedCall’conv_8/StatefulPartitionedCall’conv_9/StatefulPartitionedCall’final/StatefulPartitionedCall’"upsample_6/StatefulPartitionedCall’"upsample_7/StatefulPartitionedCall’"upsample_8/StatefulPartitionedCall’"upsample_9/StatefulPartitionedCall
#tf_op_layer_RealDiv/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????`ΐ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *Z
fURS
Q__inference_tf_op_layer_RealDiv_layer_call_and_return_conditional_losses_573782652%
#tf_op_layer_RealDiv/PartitionedCall
tf_op_layer_Sub/PartitionedCallPartitionedCall,tf_op_layer_RealDiv/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????`ΐ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *V
fQRO
M__inference_tf_op_layer_Sub_layer_call_and_return_conditional_losses_573782792!
tf_op_layer_Sub/PartitionedCallΐ
conv_1/StatefulPartitionedCallStatefulPartitionedCall(tf_op_layer_Sub/PartitionedCall:output:0conv_1_57379899conv_1_57379901*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????`ΐ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *M
fHRF
D__inference_conv_1_layer_call_and_return_conditional_losses_573782972 
conv_1/StatefulPartitionedCallΐ
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall'conv_1/StatefulPartitionedCall:output:0batch_normalization_57379904batch_normalization_57379906batch_normalization_57379908batch_normalization_57379910*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????`ΐ*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *Z
fURS
Q__inference_batch_normalization_layer_call_and_return_conditional_losses_573783502-
+batch_normalization/StatefulPartitionedCall
re_lu/PartitionedCallPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????`ΐ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *L
fGRE
C__inference_re_lu_layer_call_and_return_conditional_losses_573783912
re_lu/PartitionedCall
max_pooling2d/PartitionedCallPartitionedCallre_lu/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????0`* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *T
fORM
K__inference_max_pooling2d_layer_call_and_return_conditional_losses_573771892
max_pooling2d/PartitionedCall½
conv_2/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0conv_2_57379915conv_2_57379917*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????0` *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *M
fHRF
D__inference_conv_2_layer_call_and_return_conditional_losses_573784102 
conv_2/StatefulPartitionedCallΝ
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall'conv_2/StatefulPartitionedCall:output:0batch_normalization_1_57379920batch_normalization_1_57379922batch_normalization_1_57379924batch_normalization_1_57379926*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????0` *&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *\
fWRU
S__inference_batch_normalization_1_layer_call_and_return_conditional_losses_573784632/
-batch_normalization_1/StatefulPartitionedCall
re_lu_1/PartitionedCallPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????0` * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_re_lu_1_layer_call_and_return_conditional_losses_573785042
re_lu_1/PartitionedCall
max_pooling2d_1/PartitionedCallPartitionedCall re_lu_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????0 * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *V
fQRO
M__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_573773052!
max_pooling2d_1/PartitionedCallΏ
conv_3/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_1/PartitionedCall:output:0conv_3_57379931conv_3_57379933*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????0@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *M
fHRF
D__inference_conv_3_layer_call_and_return_conditional_losses_573785232 
conv_3/StatefulPartitionedCallΝ
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall'conv_3/StatefulPartitionedCall:output:0batch_normalization_2_57379936batch_normalization_2_57379938batch_normalization_2_57379940batch_normalization_2_57379942*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????0@*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *\
fWRU
S__inference_batch_normalization_2_layer_call_and_return_conditional_losses_573785762/
-batch_normalization_2/StatefulPartitionedCall
re_lu_2/PartitionedCallPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????0@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_re_lu_2_layer_call_and_return_conditional_losses_573786172
re_lu_2/PartitionedCall
max_pooling2d_2/PartitionedCallPartitionedCall re_lu_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *V
fQRO
M__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_573774212!
max_pooling2d_2/PartitionedCallΐ
conv_4/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_2/PartitionedCall:output:0conv_4_57379947conv_4_57379949*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *M
fHRF
D__inference_conv_4_layer_call_and_return_conditional_losses_573786362 
conv_4/StatefulPartitionedCallΞ
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCall'conv_4/StatefulPartitionedCall:output:0batch_normalization_3_57379952batch_normalization_3_57379954batch_normalization_3_57379956batch_normalization_3_57379958*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *\
fWRU
S__inference_batch_normalization_3_layer_call_and_return_conditional_losses_573786892/
-batch_normalization_3/StatefulPartitionedCall
re_lu_3/PartitionedCallPartitionedCall6batch_normalization_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_re_lu_3_layer_call_and_return_conditional_losses_573787302
re_lu_3/PartitionedCall
max_pooling2d_3/PartitionedCallPartitionedCall re_lu_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *V
fQRO
M__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_573775372!
max_pooling2d_3/PartitionedCallΐ
conv_5/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_3/PartitionedCall:output:0conv_5_57379963conv_5_57379965*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *M
fHRF
D__inference_conv_5_layer_call_and_return_conditional_losses_573787492 
conv_5/StatefulPartitionedCallΞ
-batch_normalization_4/StatefulPartitionedCallStatefulPartitionedCall'conv_5/StatefulPartitionedCall:output:0batch_normalization_4_57379968batch_normalization_4_57379970batch_normalization_4_57379972batch_normalization_4_57379974*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *\
fWRU
S__inference_batch_normalization_4_layer_call_and_return_conditional_losses_573788022/
-batch_normalization_4/StatefulPartitionedCall
re_lu_4/PartitionedCallPartitionedCall6batch_normalization_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_re_lu_4_layer_call_and_return_conditional_losses_573788432
re_lu_4/PartitionedCallή
"upsample_6/StatefulPartitionedCallStatefulPartitionedCall re_lu_4/PartitionedCall:output:0upsample_6_57379978upsample_6_57379980*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,???????????????????????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *Q
fLRJ
H__inference_upsample_6_layer_call_and_return_conditional_losses_573776852$
"upsample_6/StatefulPartitionedCallΠ
$tf_op_layer_concat_6/PartitionedCallPartitionedCall+upsample_6/StatefulPartitionedCall:output:0 re_lu_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *[
fVRT
R__inference_tf_op_layer_concat_6_layer_call_and_return_conditional_losses_573788632&
$tf_op_layer_concat_6/PartitionedCallΕ
conv_6/StatefulPartitionedCallStatefulPartitionedCall-tf_op_layer_concat_6/PartitionedCall:output:0conv_6_57379984conv_6_57379986*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *M
fHRF
D__inference_conv_6_layer_call_and_return_conditional_losses_573788822 
conv_6/StatefulPartitionedCallΞ
-batch_normalization_5/StatefulPartitionedCallStatefulPartitionedCall'conv_6/StatefulPartitionedCall:output:0batch_normalization_5_57379989batch_normalization_5_57379991batch_normalization_5_57379993batch_normalization_5_57379995*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *\
fWRU
S__inference_batch_normalization_5_layer_call_and_return_conditional_losses_573789352/
-batch_normalization_5/StatefulPartitionedCall
re_lu_5/PartitionedCallPartitionedCall6batch_normalization_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_re_lu_5_layer_call_and_return_conditional_losses_573789762
re_lu_5/PartitionedCallέ
"upsample_7/StatefulPartitionedCallStatefulPartitionedCall re_lu_5/PartitionedCall:output:0upsample_7_57379999upsample_7_57380001*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *Q
fLRJ
H__inference_upsample_7_layer_call_and_return_conditional_losses_573778372$
"upsample_7/StatefulPartitionedCallΠ
$tf_op_layer_concat_7/PartitionedCallPartitionedCall+upsample_7/StatefulPartitionedCall:output:0 re_lu_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????0* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *[
fVRT
R__inference_tf_op_layer_concat_7_layer_call_and_return_conditional_losses_573789962&
$tf_op_layer_concat_7/PartitionedCallΔ
conv_7/StatefulPartitionedCallStatefulPartitionedCall-tf_op_layer_concat_7/PartitionedCall:output:0conv_7_57380005conv_7_57380007*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????0@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *M
fHRF
D__inference_conv_7_layer_call_and_return_conditional_losses_573790152 
conv_7/StatefulPartitionedCallΝ
-batch_normalization_6/StatefulPartitionedCallStatefulPartitionedCall'conv_7/StatefulPartitionedCall:output:0batch_normalization_6_57380010batch_normalization_6_57380012batch_normalization_6_57380014batch_normalization_6_57380016*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????0@*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *\
fWRU
S__inference_batch_normalization_6_layer_call_and_return_conditional_losses_573790682/
-batch_normalization_6/StatefulPartitionedCall
re_lu_6/PartitionedCallPartitionedCall6batch_normalization_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????0@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_re_lu_6_layer_call_and_return_conditional_losses_573791092
re_lu_6/PartitionedCallέ
"upsample_8/StatefulPartitionedCallStatefulPartitionedCall re_lu_6/PartitionedCall:output:0upsample_8_57380020upsample_8_57380022*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *Q
fLRJ
H__inference_upsample_8_layer_call_and_return_conditional_losses_573779892$
"upsample_8/StatefulPartitionedCallΟ
$tf_op_layer_concat_8/PartitionedCallPartitionedCall+upsample_8/StatefulPartitionedCall:output:0 re_lu_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????0`@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *[
fVRT
R__inference_tf_op_layer_concat_8_layer_call_and_return_conditional_losses_573791292&
$tf_op_layer_concat_8/PartitionedCallΔ
conv_8/StatefulPartitionedCallStatefulPartitionedCall-tf_op_layer_concat_8/PartitionedCall:output:0conv_8_57380026conv_8_57380028*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????0` *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *M
fHRF
D__inference_conv_8_layer_call_and_return_conditional_losses_573791482 
conv_8/StatefulPartitionedCallΝ
-batch_normalization_7/StatefulPartitionedCallStatefulPartitionedCall'conv_8/StatefulPartitionedCall:output:0batch_normalization_7_57380031batch_normalization_7_57380033batch_normalization_7_57380035batch_normalization_7_57380037*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????0` *&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *\
fWRU
S__inference_batch_normalization_7_layer_call_and_return_conditional_losses_573792012/
-batch_normalization_7/StatefulPartitionedCall
re_lu_7/PartitionedCallPartitionedCall6batch_normalization_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????0` * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_re_lu_7_layer_call_and_return_conditional_losses_573792422
re_lu_7/PartitionedCallέ
"upsample_9/StatefulPartitionedCallStatefulPartitionedCall re_lu_7/PartitionedCall:output:0upsample_9_57380041upsample_9_57380043*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *Q
fLRJ
H__inference_upsample_9_layer_call_and_return_conditional_losses_573781412$
"upsample_9/StatefulPartitionedCallΞ
$tf_op_layer_concat_9/PartitionedCallPartitionedCall+upsample_9/StatefulPartitionedCall:output:0re_lu/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????`ΐ * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *[
fVRT
R__inference_tf_op_layer_concat_9_layer_call_and_return_conditional_losses_573792622&
$tf_op_layer_concat_9/PartitionedCallΕ
conv_9/StatefulPartitionedCallStatefulPartitionedCall-tf_op_layer_concat_9/PartitionedCall:output:0conv_9_57380047conv_9_57380049*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????`ΐ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *M
fHRF
D__inference_conv_9_layer_call_and_return_conditional_losses_573792812 
conv_9/StatefulPartitionedCallΞ
-batch_normalization_8/StatefulPartitionedCallStatefulPartitionedCall'conv_9/StatefulPartitionedCall:output:0batch_normalization_8_57380052batch_normalization_8_57380054batch_normalization_8_57380056batch_normalization_8_57380058*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????`ΐ*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *\
fWRU
S__inference_batch_normalization_8_layer_call_and_return_conditional_losses_573793342/
-batch_normalization_8/StatefulPartitionedCall
re_lu_8/PartitionedCallPartitionedCall6batch_normalization_8/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????`ΐ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_re_lu_8_layer_call_and_return_conditional_losses_573793752
re_lu_8/PartitionedCall³
final/StatefulPartitionedCallStatefulPartitionedCall re_lu_8/PartitionedCall:output:0final_57380062final_57380064*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????`ΐ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *L
fGRE
C__inference_final_layer_call_and_return_conditional_losses_573793932
final/StatefulPartitionedCall
IdentityIdentity&final/StatefulPartitionedCall:output:0,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall.^batch_normalization_3/StatefulPartitionedCall.^batch_normalization_4/StatefulPartitionedCall.^batch_normalization_5/StatefulPartitionedCall.^batch_normalization_6/StatefulPartitionedCall.^batch_normalization_7/StatefulPartitionedCall.^batch_normalization_8/StatefulPartitionedCall^conv_1/StatefulPartitionedCall^conv_2/StatefulPartitionedCall^conv_3/StatefulPartitionedCall^conv_4/StatefulPartitionedCall^conv_5/StatefulPartitionedCall^conv_6/StatefulPartitionedCall^conv_7/StatefulPartitionedCall^conv_8/StatefulPartitionedCall^conv_9/StatefulPartitionedCall^final/StatefulPartitionedCall#^upsample_6/StatefulPartitionedCall#^upsample_7/StatefulPartitionedCall#^upsample_8/StatefulPartitionedCall#^upsample_9/StatefulPartitionedCall*
T0*0
_output_shapes
:?????????`ΐ2

Identity"
identityIdentity:output:0*±
_input_shapes
:?????????`ΐ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2^
-batch_normalization_3/StatefulPartitionedCall-batch_normalization_3/StatefulPartitionedCall2^
-batch_normalization_4/StatefulPartitionedCall-batch_normalization_4/StatefulPartitionedCall2^
-batch_normalization_5/StatefulPartitionedCall-batch_normalization_5/StatefulPartitionedCall2^
-batch_normalization_6/StatefulPartitionedCall-batch_normalization_6/StatefulPartitionedCall2^
-batch_normalization_7/StatefulPartitionedCall-batch_normalization_7/StatefulPartitionedCall2^
-batch_normalization_8/StatefulPartitionedCall-batch_normalization_8/StatefulPartitionedCall2@
conv_1/StatefulPartitionedCallconv_1/StatefulPartitionedCall2@
conv_2/StatefulPartitionedCallconv_2/StatefulPartitionedCall2@
conv_3/StatefulPartitionedCallconv_3/StatefulPartitionedCall2@
conv_4/StatefulPartitionedCallconv_4/StatefulPartitionedCall2@
conv_5/StatefulPartitionedCallconv_5/StatefulPartitionedCall2@
conv_6/StatefulPartitionedCallconv_6/StatefulPartitionedCall2@
conv_7/StatefulPartitionedCallconv_7/StatefulPartitionedCall2@
conv_8/StatefulPartitionedCallconv_8/StatefulPartitionedCall2@
conv_9/StatefulPartitionedCallconv_9/StatefulPartitionedCall2>
final/StatefulPartitionedCallfinal/StatefulPartitionedCall2H
"upsample_6/StatefulPartitionedCall"upsample_6/StatefulPartitionedCall2H
"upsample_7/StatefulPartitionedCall"upsample_7/StatefulPartitionedCall2H
"upsample_8/StatefulPartitionedCall"upsample_8/StatefulPartitionedCall2H
"upsample_9/StatefulPartitionedCall"upsample_9/StatefulPartitionedCall:X T
0
_output_shapes
:?????????`ΐ
 
_user_specified_nameinputs
 ~
η
!__inference__traced_save_57382921
file_prefix,
(savev2_conv_1_kernel_read_readvariableop*
&savev2_conv_1_bias_read_readvariableop8
4savev2_batch_normalization_gamma_read_readvariableop7
3savev2_batch_normalization_beta_read_readvariableop>
:savev2_batch_normalization_moving_mean_read_readvariableopB
>savev2_batch_normalization_moving_variance_read_readvariableop,
(savev2_conv_2_kernel_read_readvariableop*
&savev2_conv_2_bias_read_readvariableop:
6savev2_batch_normalization_1_gamma_read_readvariableop9
5savev2_batch_normalization_1_beta_read_readvariableop@
<savev2_batch_normalization_1_moving_mean_read_readvariableopD
@savev2_batch_normalization_1_moving_variance_read_readvariableop,
(savev2_conv_3_kernel_read_readvariableop*
&savev2_conv_3_bias_read_readvariableop:
6savev2_batch_normalization_2_gamma_read_readvariableop9
5savev2_batch_normalization_2_beta_read_readvariableop@
<savev2_batch_normalization_2_moving_mean_read_readvariableopD
@savev2_batch_normalization_2_moving_variance_read_readvariableop,
(savev2_conv_4_kernel_read_readvariableop*
&savev2_conv_4_bias_read_readvariableop:
6savev2_batch_normalization_3_gamma_read_readvariableop9
5savev2_batch_normalization_3_beta_read_readvariableop@
<savev2_batch_normalization_3_moving_mean_read_readvariableopD
@savev2_batch_normalization_3_moving_variance_read_readvariableop,
(savev2_conv_5_kernel_read_readvariableop*
&savev2_conv_5_bias_read_readvariableop:
6savev2_batch_normalization_4_gamma_read_readvariableop9
5savev2_batch_normalization_4_beta_read_readvariableop@
<savev2_batch_normalization_4_moving_mean_read_readvariableopD
@savev2_batch_normalization_4_moving_variance_read_readvariableop0
,savev2_upsample_6_kernel_read_readvariableop.
*savev2_upsample_6_bias_read_readvariableop,
(savev2_conv_6_kernel_read_readvariableop*
&savev2_conv_6_bias_read_readvariableop:
6savev2_batch_normalization_5_gamma_read_readvariableop9
5savev2_batch_normalization_5_beta_read_readvariableop@
<savev2_batch_normalization_5_moving_mean_read_readvariableopD
@savev2_batch_normalization_5_moving_variance_read_readvariableop0
,savev2_upsample_7_kernel_read_readvariableop.
*savev2_upsample_7_bias_read_readvariableop,
(savev2_conv_7_kernel_read_readvariableop*
&savev2_conv_7_bias_read_readvariableop:
6savev2_batch_normalization_6_gamma_read_readvariableop9
5savev2_batch_normalization_6_beta_read_readvariableop@
<savev2_batch_normalization_6_moving_mean_read_readvariableopD
@savev2_batch_normalization_6_moving_variance_read_readvariableop0
,savev2_upsample_8_kernel_read_readvariableop.
*savev2_upsample_8_bias_read_readvariableop,
(savev2_conv_8_kernel_read_readvariableop*
&savev2_conv_8_bias_read_readvariableop:
6savev2_batch_normalization_7_gamma_read_readvariableop9
5savev2_batch_normalization_7_beta_read_readvariableop@
<savev2_batch_normalization_7_moving_mean_read_readvariableopD
@savev2_batch_normalization_7_moving_variance_read_readvariableop0
,savev2_upsample_9_kernel_read_readvariableop.
*savev2_upsample_9_bias_read_readvariableop,
(savev2_conv_9_kernel_read_readvariableop*
&savev2_conv_9_bias_read_readvariableop:
6savev2_batch_normalization_8_gamma_read_readvariableop9
5savev2_batch_normalization_8_beta_read_readvariableop@
<savev2_batch_normalization_8_moving_mean_read_readvariableopD
@savev2_batch_normalization_8_moving_variance_read_readvariableop+
'savev2_final_kernel_read_readvariableop)
%savev2_final_bias_read_readvariableop
savev2_const

identity_1’MergeV2Checkpoints
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
Const
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*<
value3B1 B+_temp_f927d7f43b004f14b7e64a69d4847ae5/part2	
Const_1
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
ShardedFilename/shard¦
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:A*
dtype0*
valueBAB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-7/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-7/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-9/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-9/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-9/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-12/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-12/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-12/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-12/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-13/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-13/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-14/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-14/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-15/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-15/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-15/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-15/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-16/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-16/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-17/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-17/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-18/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-18/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-18/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-18/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-19/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-19/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-20/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-20/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-21/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-21/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-21/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-21/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-22/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-22/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:A*
dtype0*
valueBAB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slicesσ
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0(savev2_conv_1_kernel_read_readvariableop&savev2_conv_1_bias_read_readvariableop4savev2_batch_normalization_gamma_read_readvariableop3savev2_batch_normalization_beta_read_readvariableop:savev2_batch_normalization_moving_mean_read_readvariableop>savev2_batch_normalization_moving_variance_read_readvariableop(savev2_conv_2_kernel_read_readvariableop&savev2_conv_2_bias_read_readvariableop6savev2_batch_normalization_1_gamma_read_readvariableop5savev2_batch_normalization_1_beta_read_readvariableop<savev2_batch_normalization_1_moving_mean_read_readvariableop@savev2_batch_normalization_1_moving_variance_read_readvariableop(savev2_conv_3_kernel_read_readvariableop&savev2_conv_3_bias_read_readvariableop6savev2_batch_normalization_2_gamma_read_readvariableop5savev2_batch_normalization_2_beta_read_readvariableop<savev2_batch_normalization_2_moving_mean_read_readvariableop@savev2_batch_normalization_2_moving_variance_read_readvariableop(savev2_conv_4_kernel_read_readvariableop&savev2_conv_4_bias_read_readvariableop6savev2_batch_normalization_3_gamma_read_readvariableop5savev2_batch_normalization_3_beta_read_readvariableop<savev2_batch_normalization_3_moving_mean_read_readvariableop@savev2_batch_normalization_3_moving_variance_read_readvariableop(savev2_conv_5_kernel_read_readvariableop&savev2_conv_5_bias_read_readvariableop6savev2_batch_normalization_4_gamma_read_readvariableop5savev2_batch_normalization_4_beta_read_readvariableop<savev2_batch_normalization_4_moving_mean_read_readvariableop@savev2_batch_normalization_4_moving_variance_read_readvariableop,savev2_upsample_6_kernel_read_readvariableop*savev2_upsample_6_bias_read_readvariableop(savev2_conv_6_kernel_read_readvariableop&savev2_conv_6_bias_read_readvariableop6savev2_batch_normalization_5_gamma_read_readvariableop5savev2_batch_normalization_5_beta_read_readvariableop<savev2_batch_normalization_5_moving_mean_read_readvariableop@savev2_batch_normalization_5_moving_variance_read_readvariableop,savev2_upsample_7_kernel_read_readvariableop*savev2_upsample_7_bias_read_readvariableop(savev2_conv_7_kernel_read_readvariableop&savev2_conv_7_bias_read_readvariableop6savev2_batch_normalization_6_gamma_read_readvariableop5savev2_batch_normalization_6_beta_read_readvariableop<savev2_batch_normalization_6_moving_mean_read_readvariableop@savev2_batch_normalization_6_moving_variance_read_readvariableop,savev2_upsample_8_kernel_read_readvariableop*savev2_upsample_8_bias_read_readvariableop(savev2_conv_8_kernel_read_readvariableop&savev2_conv_8_bias_read_readvariableop6savev2_batch_normalization_7_gamma_read_readvariableop5savev2_batch_normalization_7_beta_read_readvariableop<savev2_batch_normalization_7_moving_mean_read_readvariableop@savev2_batch_normalization_7_moving_variance_read_readvariableop,savev2_upsample_9_kernel_read_readvariableop*savev2_upsample_9_bias_read_readvariableop(savev2_conv_9_kernel_read_readvariableop&savev2_conv_9_bias_read_readvariableop6savev2_batch_normalization_8_gamma_read_readvariableop5savev2_batch_normalization_8_beta_read_readvariableop<savev2_batch_normalization_8_moving_mean_read_readvariableop@savev2_batch_normalization_8_moving_variance_read_readvariableop'savev2_final_kernel_read_readvariableop%savev2_final_bias_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *O
dtypesE
C2A2
SaveV2Ί
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes‘
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

identity_1Identity_1:output:0*Ϊ
_input_shapesΘ
Ε: ::::::: : : : : : : @:@:@:@:@:@:@::::::::::::::::::::@:@:@:@:@:@:@:@: @: :@ : : : : : : :: :::::::: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:,(
&
_output_shapes
:: 
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
::,(
&
_output_shapes
: : 
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
: :,(
&
_output_shapes
: @: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@:-)
'
_output_shapes
:@:!

_output_shapes	
::!

_output_shapes	
::!

_output_shapes	
::!

_output_shapes	
::!

_output_shapes	
::.*
(
_output_shapes
::!

_output_shapes	
::!

_output_shapes	
::!

_output_shapes	
::!

_output_shapes	
::!

_output_shapes	
::.*
(
_output_shapes
::! 

_output_shapes	
::.!*
(
_output_shapes
::!"

_output_shapes	
::!#

_output_shapes	
::!$

_output_shapes	
::!%

_output_shapes	
::!&

_output_shapes	
::-')
'
_output_shapes
:@: (

_output_shapes
:@:-))
'
_output_shapes
:@: *

_output_shapes
:@: +

_output_shapes
:@: ,

_output_shapes
:@: -

_output_shapes
:@: .

_output_shapes
:@:,/(
&
_output_shapes
: @: 0

_output_shapes
: :,1(
&
_output_shapes
:@ : 2

_output_shapes
: : 3

_output_shapes
: : 4

_output_shapes
: : 5

_output_shapes
: : 6

_output_shapes
: :,7(
&
_output_shapes
: : 8

_output_shapes
::,9(
&
_output_shapes
: : :

_output_shapes
:: ;

_output_shapes
:: <

_output_shapes
:: =

_output_shapes
:: >

_output_shapes
::,?(
&
_output_shapes
:: @

_output_shapes
::A

_output_shapes
: 
Σ
a
E__inference_re_lu_7_layer_call_and_return_conditional_losses_57379242

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:?????????0` 2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:?????????0` 2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????0` :W S
/
_output_shapes
:?????????0` 
 
_user_specified_nameinputs
β
«
8__inference_batch_normalization_3_layer_call_fn_57381827

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity’StatefulPartitionedCall©
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *\
fWRU
S__inference_batch_normalization_3_layer_call_and_return_conditional_losses_573786712
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:?????????::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:?????????
 
_user_specified_nameinputs
Χ
a
E__inference_re_lu_4_layer_call_and_return_conditional_losses_57382002

inputs
identityW
ReluReluinputs*
T0*0
_output_shapes
:?????????2
Reluo
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:?????????:X T
0
_output_shapes
:?????????
 
_user_specified_nameinputs
Λ
°
S__inference_batch_normalization_1_layer_call_and_return_conditional_losses_57381418

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity’AssignNewValue’AssignNewValue_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1κ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+??????????????????????????? : : : : :*
epsilon%o:*
exponential_avg_factor%
Χ#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1¦
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+??????????????????????????? ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
₯
¬
D__inference_conv_8_layer_call_and_return_conditional_losses_57379148

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@ *
dtype02
Conv2D/ReadVariableOp£
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????0` *
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????0` 2	
BiasAddl
IdentityIdentityBiasAdd:output:0*
T0*/
_output_shapes
:?????????0` 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????0`@:::W S
/
_output_shapes
:?????????0`@
 
_user_specified_nameinputs

i
M__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_57377305

inputs
identity­
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
2	
MaxPool
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs

c
7__inference_tf_op_layer_concat_7_layer_call_fn_57382190
inputs_0
inputs_1
identityλ
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????0* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *[
fVRT
R__inference_tf_op_layer_concat_7_layer_call_and_return_conditional_losses_573789962
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:?????????02

Identity"
identityIdentity:output:0*[
_input_shapesJ
H:+???????????????????????????@:?????????0@:k g
A
_output_shapes/
-:+???????????????????????????@
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:?????????0@
"
_user_specified_name
inputs/1
Φ

-__inference_upsample_6_layer_call_fn_57377695

inputs
unknown
	unknown_0
identity’StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,???????????????????????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *Q
fLRJ
H__inference_upsample_6_layer_call_and_return_conditional_losses_573776852
StatefulPartitionedCall©
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,???????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:,???????????????????????????::22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,???????????????????????????
 
_user_specified_nameinputs

c
7__inference_tf_op_layer_concat_9_layer_call_fn_57382530
inputs_0
inputs_1
identityλ
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????`ΐ * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *[
fVRT
R__inference_tf_op_layer_concat_9_layer_call_and_return_conditional_losses_573792622
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:?????????`ΐ 2

Identity"
identityIdentity:output:0*\
_input_shapesK
I:+???????????????????????????:?????????`ΐ:k g
A
_output_shapes/
-:+???????????????????????????
"
_user_specified_name
inputs/0:ZV
0
_output_shapes
:?????????`ΐ
"
_user_specified_name
inputs/1

~
)__inference_conv_6_layer_call_fn_57382039

inputs
unknown
	unknown_0
identity’StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *M
fHRF
D__inference_conv_6_layer_call_and_return_conditional_losses_573788822
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:?????????::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:?????????
 
_user_specified_nameinputs
Χ
°
S__inference_batch_normalization_3_layer_call_and_return_conditional_losses_57381732

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity’AssignNewValue’AssignNewValue_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype02
ReadVariableOp_1¨
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ο
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,???????????????????????????:::::*
epsilon%o:*
exponential_avg_factor%
Χ#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1§
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*B
_output_shapes0
.:,???????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,???????????????????????????::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:j f
B
_output_shapes0
.:,???????????????????????????
 
_user_specified_nameinputs
­
¬
D__inference_conv_5_layer_call_and_return_conditional_losses_57381860

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp€
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????2	
BiasAddm
IdentityIdentityBiasAdd:output:0*
T0*0
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:?????????:::X T
0
_output_shapes
:?????????
 
_user_specified_nameinputs
Ν
N
2__inference_tf_op_layer_Sub_layer_call_fn_57381222

inputs
identityΩ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????`ΐ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *V
fQRO
M__inference_tf_op_layer_Sub_layer_call_and_return_conditional_losses_573782792
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:?????????`ΐ2

Identity"
identityIdentity:output:0*/
_input_shapes
:?????????`ΐ:X T
0
_output_shapes
:?????????`ΐ
 
_user_specified_nameinputs
?

S__inference_batch_normalization_6_layer_call_and_return_conditional_losses_57382247

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Κ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????0@:@:@:@:@:*
epsilon%o:*
is_training( 2
FusedBatchNormV3p
IdentityIdentityFusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????0@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????0@:::::W S
/
_output_shapes
:?????????0@
 
_user_specified_nameinputs
ή

S__inference_batch_normalization_5_layer_call_and_return_conditional_losses_57378935

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityu
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype02
ReadVariableOp_1¨
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ο
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:?????????:::::*
epsilon%o:*
is_training( 2
FusedBatchNormV3q
IdentityIdentityFusedBatchNormV3:y:0*
T0*0
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:?????????:::::X T
0
_output_shapes
:?????????
 
_user_specified_nameinputs
­
Ϋ
/__inference_functional_1_layer_call_fn_57381067

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

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28

unknown_29

unknown_30

unknown_31

unknown_32

unknown_33

unknown_34

unknown_35

unknown_36

unknown_37

unknown_38

unknown_39

unknown_40

unknown_41

unknown_42

unknown_43

unknown_44

unknown_45

unknown_46

unknown_47

unknown_48

unknown_49

unknown_50

unknown_51

unknown_52

unknown_53

unknown_54

unknown_55

unknown_56

unknown_57

unknown_58

unknown_59

unknown_60

unknown_61

unknown_62
identity’StatefulPartitionedCallΡ	
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46
unknown_47
unknown_48
unknown_49
unknown_50
unknown_51
unknown_52
unknown_53
unknown_54
unknown_55
unknown_56
unknown_57
unknown_58
unknown_59
unknown_60
unknown_61
unknown_62*L
TinE
C2A*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????`ΐ*P
_read_only_resource_inputs2
0.	
 !"#$'()*+,/01234789:;<?@*2
config_proto" 

CPU

GPU2 *0J 8 *S
fNRL
J__inference_functional_1_layer_call_and_return_conditional_losses_573797612
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:?????????`ΐ2

Identity"
identityIdentity:output:0*±
_input_shapes
:?????????`ΐ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:?????????`ΐ
 
_user_specified_nameinputs
Σ
a
E__inference_re_lu_1_layer_call_and_return_conditional_losses_57378504

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:?????????0` 2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:?????????0` 2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????0` :W S
/
_output_shapes
:?????????0` 
 
_user_specified_nameinputs

°
S__inference_batch_normalization_5_layer_call_and_return_conditional_losses_57378917

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity’AssignNewValue’AssignNewValue_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype02
ReadVariableOp_1¨
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1έ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:?????????:::::*
epsilon%o:*
exponential_avg_factor%
Χ#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*0
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:?????????::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:X T
0
_output_shapes
:?????????
 
_user_specified_nameinputs
¦
«
8__inference_batch_normalization_2_layer_call_fn_57381670

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity’StatefulPartitionedCallΊ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *\
fWRU
S__inference_batch_normalization_2_layer_call_and_return_conditional_losses_573773732
StatefulPartitionedCall¨
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????@::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs

~
)__inference_conv_1_layer_call_fn_57381241

inputs
unknown
	unknown_0
identity’StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????`ΐ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *M
fHRF
D__inference_conv_1_layer_call_and_return_conditional_losses_573782972
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:?????????`ΐ2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:?????????`ΐ::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:?????????`ΐ
 
_user_specified_nameinputs
ͺ
¬
D__inference_conv_4_layer_call_and_return_conditional_losses_57381703

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@*
dtype02
Conv2D/ReadVariableOp€
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????2	
BiasAddm
IdentityIdentityBiasAdd:output:0*
T0*0
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????@:::W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
Υ
_
C__inference_re_lu_layer_call_and_return_conditional_losses_57378391

inputs
identityW
ReluReluinputs*
T0*0
_output_shapes
:?????????`ΐ2
Reluo
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:?????????`ΐ2

Identity"
identityIdentity:output:0*/
_input_shapes
:?????????`ΐ:X T
0
_output_shapes
:?????????`ΐ
 
_user_specified_nameinputs
?

S__inference_batch_normalization_1_layer_call_and_return_conditional_losses_57381500

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Κ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????0` : : : : :*
epsilon%o:*
is_training( 2
FusedBatchNormV3p
IdentityIdentityFusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????0` 2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????0` :::::W S
/
_output_shapes
:?????????0` 
 
_user_specified_nameinputs
Σ
a
E__inference_re_lu_2_layer_call_and_return_conditional_losses_57381688

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:?????????0@2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:?????????0@2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????0@:W S
/
_output_shapes
:?????????0@
 
_user_specified_nameinputs
Σ
a
E__inference_re_lu_2_layer_call_and_return_conditional_losses_57378617

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:?????????0@2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:?????????0@2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????0@:W S
/
_output_shapes
:?????????0@
 
_user_specified_nameinputs
ͺ
¬
D__inference_conv_9_layer_call_and_return_conditional_losses_57379281

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOp€
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????`ΐ*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????`ΐ2	
BiasAddm
IdentityIdentityBiasAdd:output:0*
T0*0
_output_shapes
:?????????`ΐ2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:?????????`ΐ :::X T
0
_output_shapes
:?????????`ΐ 
 
_user_specified_nameinputs
Ή
F
*__inference_re_lu_7_layer_call_fn_57382517

inputs
identityΠ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????0` * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_re_lu_7_layer_call_and_return_conditional_losses_573792422
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????0` 2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????0` :W S
/
_output_shapes
:?????????0` 
 
_user_specified_nameinputs
?

S__inference_batch_normalization_7_layer_call_and_return_conditional_losses_57379201

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Κ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????0` : : : : :*
epsilon%o:*
is_training( 2
FusedBatchNormV3p
IdentityIdentityFusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????0` 2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????0` :::::W S
/
_output_shapes
:?????????0` 
 
_user_specified_nameinputs
¨
«
8__inference_batch_normalization_1_layer_call_fn_57381462

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity’StatefulPartitionedCallΌ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? *&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *\
fWRU
S__inference_batch_normalization_1_layer_call_and_return_conditional_losses_573772882
StatefulPartitionedCall¨
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+??????????????????????????? ::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs


S__inference_batch_normalization_1_layer_call_and_return_conditional_losses_57381436

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ά
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+??????????????????????????? : : : : :*
epsilon%o:*
is_training( 2
FusedBatchNormV3
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+??????????????????????????? :::::i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs


S__inference_batch_normalization_8_layer_call_and_return_conditional_losses_57378244

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ά
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o:*
is_training( 2
FusedBatchNormV3
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????:::::i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
Χ
°
S__inference_batch_normalization_4_layer_call_and_return_conditional_losses_57377605

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity’AssignNewValue’AssignNewValue_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype02
ReadVariableOp_1¨
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ο
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,???????????????????????????:::::*
epsilon%o:*
exponential_avg_factor%
Χ#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1§
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*B
_output_shapes0
.:,???????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,???????????????????????????::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:j f
B
_output_shapes0
.:,???????????????????????????
 
_user_specified_nameinputs
Χ
a
E__inference_re_lu_8_layer_call_and_return_conditional_losses_57379375

inputs
identityW
ReluReluinputs*
T0*0
_output_shapes
:?????????`ΐ2
Reluo
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:?????????`ΐ2

Identity"
identityIdentity:output:0*/
_input_shapes
:?????????`ΐ:X T
0
_output_shapes
:?????????`ΐ
 
_user_specified_nameinputs
δ
«
8__inference_batch_normalization_8_layer_call_fn_57382677

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity’StatefulPartitionedCall«
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????`ΐ*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *\
fWRU
S__inference_batch_normalization_8_layer_call_and_return_conditional_losses_573793342
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:?????????`ΐ2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:?????????`ΐ::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:?????????`ΐ
 
_user_specified_nameinputs
?

S__inference_batch_normalization_2_layer_call_and_return_conditional_losses_57381593

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Κ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????0@:@:@:@:@:*
epsilon%o:*
is_training( 2
FusedBatchNormV3p
IdentityIdentityFusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????0@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????0@:::::W S
/
_output_shapes
:?????????0@
 
_user_specified_nameinputs
Χ
°
S__inference_batch_normalization_3_layer_call_and_return_conditional_losses_57377489

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity’AssignNewValue’AssignNewValue_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype02
ReadVariableOp_1¨
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ο
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,???????????????????????????:::::*
epsilon%o:*
exponential_avg_factor%
Χ#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1§
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*B
_output_shapes0
.:,???????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,???????????????????????????::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:j f
B
_output_shapes0
.:,???????????????????????????
 
_user_specified_nameinputs


S__inference_batch_normalization_1_layer_call_and_return_conditional_losses_57377288

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ά
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+??????????????????????????? : : : : :*
epsilon%o:*
is_training( 2
FusedBatchNormV3
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+??????????????????????????? :::::i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
Ε
i
M__inference_tf_op_layer_Sub_layer_call_and_return_conditional_losses_57378279

inputs
identityS
Sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
Sub/ys
SubSubinputsSub/y:output:0*
T0*
_cloned(*0
_output_shapes
:?????????`ΐ2
Subd
IdentityIdentitySub:z:0*
T0*0
_output_shapes
:?????????`ΐ2

Identity"
identityIdentity:output:0*/
_input_shapes
:?????????`ΐ:X T
0
_output_shapes
:?????????`ΐ
 
_user_specified_nameinputs
Λ
°
S__inference_batch_normalization_7_layer_call_and_return_conditional_losses_57382399

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity’AssignNewValue’AssignNewValue_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1κ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+??????????????????????????? : : : : :*
epsilon%o:*
exponential_avg_factor%
Χ#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1¦
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+??????????????????????????? ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
β
«
8__inference_batch_normalization_5_layer_call_fn_57382090

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity’StatefulPartitionedCall©
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *\
fWRU
S__inference_batch_normalization_5_layer_call_and_return_conditional_losses_573789172
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:?????????::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:?????????
 
_user_specified_nameinputs
Λ$
Ί
H__inference_upsample_7_layer_call_and_return_conditional_losses_57377837

inputs,
(conv2d_transpose_readvariableop_resource#
biasadd_readvariableop_resource
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2β
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2μ
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1x
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2μ
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_2P
mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
mul/y\
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: 2
mulP
add/yConst*
_output_shapes
: *
dtype0*
value	B : 2
add/yM
addAddV2mul:z:0add/y:output:0*
T0*
_output_shapes
: 2
addT
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2	
mul_1/yb
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: 2
mul_1T
add_1/yConst*
_output_shapes
: *
dtype0*
value	B : 2	
add_1/yU
add_1AddV2	mul_1:z:0add_1/y:output:0*
T0*
_output_shapes
: 2
add_1T
stack/3Const*
_output_shapes
: *
dtype0*
value	B :@2	
stack/3
stackPackstrided_slice:output:0add:z:0	add_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:2
stackx
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2μ
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_3΄
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*'
_output_shapes
:@*
dtype02!
conv2d_transpose/ReadVariableOpρ
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+???????????????????????????@*
paddingVALID*
strides
2
conv2d_transpose
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp€
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????@2	
BiasAdd~
IdentityIdentityBiasAdd:output:0*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:,???????????????????????????:::j f
B
_output_shapes0
.:,???????????????????????????
 
_user_specified_nameinputs
½
F
*__inference_re_lu_5_layer_call_fn_57382177

inputs
identityΡ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_re_lu_5_layer_call_and_return_conditional_losses_573789762
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:?????????:X T
0
_output_shapes
:?????????
 
_user_specified_nameinputs
?

-__inference_upsample_9_layer_call_fn_57378151

inputs
unknown
	unknown_0
identity’StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *Q
fLRJ
H__inference_upsample_9_layer_call_and_return_conditional_losses_573781412
StatefulPartitionedCall¨
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+??????????????????????????? ::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
₯
¬
D__inference_conv_2_layer_call_and_return_conditional_losses_57378410

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOp£
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????0` *
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????0` 2	
BiasAddl
IdentityIdentityBiasAdd:output:0*
T0*/
_output_shapes
:?????????0` 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????0`:::W S
/
_output_shapes
:?????????0`
 
_user_specified_nameinputs
ͺ
«
8__inference_batch_normalization_5_layer_call_fn_57382154

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity’StatefulPartitionedCall»
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,???????????????????????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *\
fWRU
S__inference_batch_normalization_5_layer_call_and_return_conditional_losses_573777572
StatefulPartitionedCall©
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,???????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,???????????????????????????::::22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,???????????????????????????
 
_user_specified_nameinputs
₯
¬
D__inference_conv_3_layer_call_and_return_conditional_losses_57378523

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02
Conv2D/ReadVariableOp£
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????0@*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????0@2	
BiasAddl
IdentityIdentityBiasAdd:output:0*
T0*/
_output_shapes
:?????????0@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????0 :::W S
/
_output_shapes
:?????????0 
 
_user_specified_nameinputs
Κ
κ
J__inference_functional_1_layer_call_and_return_conditional_losses_57379584

imageinput
conv_1_57379415
conv_1_57379417 
batch_normalization_57379420 
batch_normalization_57379422 
batch_normalization_57379424 
batch_normalization_57379426
conv_2_57379431
conv_2_57379433"
batch_normalization_1_57379436"
batch_normalization_1_57379438"
batch_normalization_1_57379440"
batch_normalization_1_57379442
conv_3_57379447
conv_3_57379449"
batch_normalization_2_57379452"
batch_normalization_2_57379454"
batch_normalization_2_57379456"
batch_normalization_2_57379458
conv_4_57379463
conv_4_57379465"
batch_normalization_3_57379468"
batch_normalization_3_57379470"
batch_normalization_3_57379472"
batch_normalization_3_57379474
conv_5_57379479
conv_5_57379481"
batch_normalization_4_57379484"
batch_normalization_4_57379486"
batch_normalization_4_57379488"
batch_normalization_4_57379490
upsample_6_57379494
upsample_6_57379496
conv_6_57379500
conv_6_57379502"
batch_normalization_5_57379505"
batch_normalization_5_57379507"
batch_normalization_5_57379509"
batch_normalization_5_57379511
upsample_7_57379515
upsample_7_57379517
conv_7_57379521
conv_7_57379523"
batch_normalization_6_57379526"
batch_normalization_6_57379528"
batch_normalization_6_57379530"
batch_normalization_6_57379532
upsample_8_57379536
upsample_8_57379538
conv_8_57379542
conv_8_57379544"
batch_normalization_7_57379547"
batch_normalization_7_57379549"
batch_normalization_7_57379551"
batch_normalization_7_57379553
upsample_9_57379557
upsample_9_57379559
conv_9_57379563
conv_9_57379565"
batch_normalization_8_57379568"
batch_normalization_8_57379570"
batch_normalization_8_57379572"
batch_normalization_8_57379574
final_57379578
final_57379580
identity’+batch_normalization/StatefulPartitionedCall’-batch_normalization_1/StatefulPartitionedCall’-batch_normalization_2/StatefulPartitionedCall’-batch_normalization_3/StatefulPartitionedCall’-batch_normalization_4/StatefulPartitionedCall’-batch_normalization_5/StatefulPartitionedCall’-batch_normalization_6/StatefulPartitionedCall’-batch_normalization_7/StatefulPartitionedCall’-batch_normalization_8/StatefulPartitionedCall’conv_1/StatefulPartitionedCall’conv_2/StatefulPartitionedCall’conv_3/StatefulPartitionedCall’conv_4/StatefulPartitionedCall’conv_5/StatefulPartitionedCall’conv_6/StatefulPartitionedCall’conv_7/StatefulPartitionedCall’conv_8/StatefulPartitionedCall’conv_9/StatefulPartitionedCall’final/StatefulPartitionedCall’"upsample_6/StatefulPartitionedCall’"upsample_7/StatefulPartitionedCall’"upsample_8/StatefulPartitionedCall’"upsample_9/StatefulPartitionedCall
#tf_op_layer_RealDiv/PartitionedCallPartitionedCall
imageinput*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????`ΐ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *Z
fURS
Q__inference_tf_op_layer_RealDiv_layer_call_and_return_conditional_losses_573782652%
#tf_op_layer_RealDiv/PartitionedCall
tf_op_layer_Sub/PartitionedCallPartitionedCall,tf_op_layer_RealDiv/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????`ΐ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *V
fQRO
M__inference_tf_op_layer_Sub_layer_call_and_return_conditional_losses_573782792!
tf_op_layer_Sub/PartitionedCallΐ
conv_1/StatefulPartitionedCallStatefulPartitionedCall(tf_op_layer_Sub/PartitionedCall:output:0conv_1_57379415conv_1_57379417*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????`ΐ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *M
fHRF
D__inference_conv_1_layer_call_and_return_conditional_losses_573782972 
conv_1/StatefulPartitionedCallΐ
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall'conv_1/StatefulPartitionedCall:output:0batch_normalization_57379420batch_normalization_57379422batch_normalization_57379424batch_normalization_57379426*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????`ΐ*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *Z
fURS
Q__inference_batch_normalization_layer_call_and_return_conditional_losses_573783502-
+batch_normalization/StatefulPartitionedCall
re_lu/PartitionedCallPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????`ΐ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *L
fGRE
C__inference_re_lu_layer_call_and_return_conditional_losses_573783912
re_lu/PartitionedCall
max_pooling2d/PartitionedCallPartitionedCallre_lu/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????0`* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *T
fORM
K__inference_max_pooling2d_layer_call_and_return_conditional_losses_573771892
max_pooling2d/PartitionedCall½
conv_2/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0conv_2_57379431conv_2_57379433*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????0` *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *M
fHRF
D__inference_conv_2_layer_call_and_return_conditional_losses_573784102 
conv_2/StatefulPartitionedCallΝ
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall'conv_2/StatefulPartitionedCall:output:0batch_normalization_1_57379436batch_normalization_1_57379438batch_normalization_1_57379440batch_normalization_1_57379442*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????0` *&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *\
fWRU
S__inference_batch_normalization_1_layer_call_and_return_conditional_losses_573784632/
-batch_normalization_1/StatefulPartitionedCall
re_lu_1/PartitionedCallPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????0` * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_re_lu_1_layer_call_and_return_conditional_losses_573785042
re_lu_1/PartitionedCall
max_pooling2d_1/PartitionedCallPartitionedCall re_lu_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????0 * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *V
fQRO
M__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_573773052!
max_pooling2d_1/PartitionedCallΏ
conv_3/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_1/PartitionedCall:output:0conv_3_57379447conv_3_57379449*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????0@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *M
fHRF
D__inference_conv_3_layer_call_and_return_conditional_losses_573785232 
conv_3/StatefulPartitionedCallΝ
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall'conv_3/StatefulPartitionedCall:output:0batch_normalization_2_57379452batch_normalization_2_57379454batch_normalization_2_57379456batch_normalization_2_57379458*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????0@*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *\
fWRU
S__inference_batch_normalization_2_layer_call_and_return_conditional_losses_573785762/
-batch_normalization_2/StatefulPartitionedCall
re_lu_2/PartitionedCallPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????0@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_re_lu_2_layer_call_and_return_conditional_losses_573786172
re_lu_2/PartitionedCall
max_pooling2d_2/PartitionedCallPartitionedCall re_lu_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *V
fQRO
M__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_573774212!
max_pooling2d_2/PartitionedCallΐ
conv_4/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_2/PartitionedCall:output:0conv_4_57379463conv_4_57379465*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *M
fHRF
D__inference_conv_4_layer_call_and_return_conditional_losses_573786362 
conv_4/StatefulPartitionedCallΞ
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCall'conv_4/StatefulPartitionedCall:output:0batch_normalization_3_57379468batch_normalization_3_57379470batch_normalization_3_57379472batch_normalization_3_57379474*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *\
fWRU
S__inference_batch_normalization_3_layer_call_and_return_conditional_losses_573786892/
-batch_normalization_3/StatefulPartitionedCall
re_lu_3/PartitionedCallPartitionedCall6batch_normalization_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_re_lu_3_layer_call_and_return_conditional_losses_573787302
re_lu_3/PartitionedCall
max_pooling2d_3/PartitionedCallPartitionedCall re_lu_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *V
fQRO
M__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_573775372!
max_pooling2d_3/PartitionedCallΐ
conv_5/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_3/PartitionedCall:output:0conv_5_57379479conv_5_57379481*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *M
fHRF
D__inference_conv_5_layer_call_and_return_conditional_losses_573787492 
conv_5/StatefulPartitionedCallΞ
-batch_normalization_4/StatefulPartitionedCallStatefulPartitionedCall'conv_5/StatefulPartitionedCall:output:0batch_normalization_4_57379484batch_normalization_4_57379486batch_normalization_4_57379488batch_normalization_4_57379490*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *\
fWRU
S__inference_batch_normalization_4_layer_call_and_return_conditional_losses_573788022/
-batch_normalization_4/StatefulPartitionedCall
re_lu_4/PartitionedCallPartitionedCall6batch_normalization_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_re_lu_4_layer_call_and_return_conditional_losses_573788432
re_lu_4/PartitionedCallή
"upsample_6/StatefulPartitionedCallStatefulPartitionedCall re_lu_4/PartitionedCall:output:0upsample_6_57379494upsample_6_57379496*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,???????????????????????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *Q
fLRJ
H__inference_upsample_6_layer_call_and_return_conditional_losses_573776852$
"upsample_6/StatefulPartitionedCallΠ
$tf_op_layer_concat_6/PartitionedCallPartitionedCall+upsample_6/StatefulPartitionedCall:output:0 re_lu_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *[
fVRT
R__inference_tf_op_layer_concat_6_layer_call_and_return_conditional_losses_573788632&
$tf_op_layer_concat_6/PartitionedCallΕ
conv_6/StatefulPartitionedCallStatefulPartitionedCall-tf_op_layer_concat_6/PartitionedCall:output:0conv_6_57379500conv_6_57379502*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *M
fHRF
D__inference_conv_6_layer_call_and_return_conditional_losses_573788822 
conv_6/StatefulPartitionedCallΞ
-batch_normalization_5/StatefulPartitionedCallStatefulPartitionedCall'conv_6/StatefulPartitionedCall:output:0batch_normalization_5_57379505batch_normalization_5_57379507batch_normalization_5_57379509batch_normalization_5_57379511*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *\
fWRU
S__inference_batch_normalization_5_layer_call_and_return_conditional_losses_573789352/
-batch_normalization_5/StatefulPartitionedCall
re_lu_5/PartitionedCallPartitionedCall6batch_normalization_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_re_lu_5_layer_call_and_return_conditional_losses_573789762
re_lu_5/PartitionedCallέ
"upsample_7/StatefulPartitionedCallStatefulPartitionedCall re_lu_5/PartitionedCall:output:0upsample_7_57379515upsample_7_57379517*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *Q
fLRJ
H__inference_upsample_7_layer_call_and_return_conditional_losses_573778372$
"upsample_7/StatefulPartitionedCallΠ
$tf_op_layer_concat_7/PartitionedCallPartitionedCall+upsample_7/StatefulPartitionedCall:output:0 re_lu_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????0* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *[
fVRT
R__inference_tf_op_layer_concat_7_layer_call_and_return_conditional_losses_573789962&
$tf_op_layer_concat_7/PartitionedCallΔ
conv_7/StatefulPartitionedCallStatefulPartitionedCall-tf_op_layer_concat_7/PartitionedCall:output:0conv_7_57379521conv_7_57379523*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????0@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *M
fHRF
D__inference_conv_7_layer_call_and_return_conditional_losses_573790152 
conv_7/StatefulPartitionedCallΝ
-batch_normalization_6/StatefulPartitionedCallStatefulPartitionedCall'conv_7/StatefulPartitionedCall:output:0batch_normalization_6_57379526batch_normalization_6_57379528batch_normalization_6_57379530batch_normalization_6_57379532*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????0@*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *\
fWRU
S__inference_batch_normalization_6_layer_call_and_return_conditional_losses_573790682/
-batch_normalization_6/StatefulPartitionedCall
re_lu_6/PartitionedCallPartitionedCall6batch_normalization_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????0@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_re_lu_6_layer_call_and_return_conditional_losses_573791092
re_lu_6/PartitionedCallέ
"upsample_8/StatefulPartitionedCallStatefulPartitionedCall re_lu_6/PartitionedCall:output:0upsample_8_57379536upsample_8_57379538*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *Q
fLRJ
H__inference_upsample_8_layer_call_and_return_conditional_losses_573779892$
"upsample_8/StatefulPartitionedCallΟ
$tf_op_layer_concat_8/PartitionedCallPartitionedCall+upsample_8/StatefulPartitionedCall:output:0 re_lu_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????0`@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *[
fVRT
R__inference_tf_op_layer_concat_8_layer_call_and_return_conditional_losses_573791292&
$tf_op_layer_concat_8/PartitionedCallΔ
conv_8/StatefulPartitionedCallStatefulPartitionedCall-tf_op_layer_concat_8/PartitionedCall:output:0conv_8_57379542conv_8_57379544*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????0` *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *M
fHRF
D__inference_conv_8_layer_call_and_return_conditional_losses_573791482 
conv_8/StatefulPartitionedCallΝ
-batch_normalization_7/StatefulPartitionedCallStatefulPartitionedCall'conv_8/StatefulPartitionedCall:output:0batch_normalization_7_57379547batch_normalization_7_57379549batch_normalization_7_57379551batch_normalization_7_57379553*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????0` *&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *\
fWRU
S__inference_batch_normalization_7_layer_call_and_return_conditional_losses_573792012/
-batch_normalization_7/StatefulPartitionedCall
re_lu_7/PartitionedCallPartitionedCall6batch_normalization_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????0` * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_re_lu_7_layer_call_and_return_conditional_losses_573792422
re_lu_7/PartitionedCallέ
"upsample_9/StatefulPartitionedCallStatefulPartitionedCall re_lu_7/PartitionedCall:output:0upsample_9_57379557upsample_9_57379559*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *Q
fLRJ
H__inference_upsample_9_layer_call_and_return_conditional_losses_573781412$
"upsample_9/StatefulPartitionedCallΞ
$tf_op_layer_concat_9/PartitionedCallPartitionedCall+upsample_9/StatefulPartitionedCall:output:0re_lu/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????`ΐ * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *[
fVRT
R__inference_tf_op_layer_concat_9_layer_call_and_return_conditional_losses_573792622&
$tf_op_layer_concat_9/PartitionedCallΕ
conv_9/StatefulPartitionedCallStatefulPartitionedCall-tf_op_layer_concat_9/PartitionedCall:output:0conv_9_57379563conv_9_57379565*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????`ΐ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *M
fHRF
D__inference_conv_9_layer_call_and_return_conditional_losses_573792812 
conv_9/StatefulPartitionedCallΞ
-batch_normalization_8/StatefulPartitionedCallStatefulPartitionedCall'conv_9/StatefulPartitionedCall:output:0batch_normalization_8_57379568batch_normalization_8_57379570batch_normalization_8_57379572batch_normalization_8_57379574*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????`ΐ*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *\
fWRU
S__inference_batch_normalization_8_layer_call_and_return_conditional_losses_573793342/
-batch_normalization_8/StatefulPartitionedCall
re_lu_8/PartitionedCallPartitionedCall6batch_normalization_8/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????`ΐ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_re_lu_8_layer_call_and_return_conditional_losses_573793752
re_lu_8/PartitionedCall³
final/StatefulPartitionedCallStatefulPartitionedCall re_lu_8/PartitionedCall:output:0final_57379578final_57379580*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????`ΐ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *L
fGRE
C__inference_final_layer_call_and_return_conditional_losses_573793932
final/StatefulPartitionedCall
IdentityIdentity&final/StatefulPartitionedCall:output:0,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall.^batch_normalization_3/StatefulPartitionedCall.^batch_normalization_4/StatefulPartitionedCall.^batch_normalization_5/StatefulPartitionedCall.^batch_normalization_6/StatefulPartitionedCall.^batch_normalization_7/StatefulPartitionedCall.^batch_normalization_8/StatefulPartitionedCall^conv_1/StatefulPartitionedCall^conv_2/StatefulPartitionedCall^conv_3/StatefulPartitionedCall^conv_4/StatefulPartitionedCall^conv_5/StatefulPartitionedCall^conv_6/StatefulPartitionedCall^conv_7/StatefulPartitionedCall^conv_8/StatefulPartitionedCall^conv_9/StatefulPartitionedCall^final/StatefulPartitionedCall#^upsample_6/StatefulPartitionedCall#^upsample_7/StatefulPartitionedCall#^upsample_8/StatefulPartitionedCall#^upsample_9/StatefulPartitionedCall*
T0*0
_output_shapes
:?????????`ΐ2

Identity"
identityIdentity:output:0*±
_input_shapes
:?????????`ΐ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2^
-batch_normalization_3/StatefulPartitionedCall-batch_normalization_3/StatefulPartitionedCall2^
-batch_normalization_4/StatefulPartitionedCall-batch_normalization_4/StatefulPartitionedCall2^
-batch_normalization_5/StatefulPartitionedCall-batch_normalization_5/StatefulPartitionedCall2^
-batch_normalization_6/StatefulPartitionedCall-batch_normalization_6/StatefulPartitionedCall2^
-batch_normalization_7/StatefulPartitionedCall-batch_normalization_7/StatefulPartitionedCall2^
-batch_normalization_8/StatefulPartitionedCall-batch_normalization_8/StatefulPartitionedCall2@
conv_1/StatefulPartitionedCallconv_1/StatefulPartitionedCall2@
conv_2/StatefulPartitionedCallconv_2/StatefulPartitionedCall2@
conv_3/StatefulPartitionedCallconv_3/StatefulPartitionedCall2@
conv_4/StatefulPartitionedCallconv_4/StatefulPartitionedCall2@
conv_5/StatefulPartitionedCallconv_5/StatefulPartitionedCall2@
conv_6/StatefulPartitionedCallconv_6/StatefulPartitionedCall2@
conv_7/StatefulPartitionedCallconv_7/StatefulPartitionedCall2@
conv_8/StatefulPartitionedCallconv_8/StatefulPartitionedCall2@
conv_9/StatefulPartitionedCallconv_9/StatefulPartitionedCall2>
final/StatefulPartitionedCallfinal/StatefulPartitionedCall2H
"upsample_6/StatefulPartitionedCall"upsample_6/StatefulPartitionedCall2H
"upsample_7/StatefulPartitionedCall"upsample_7/StatefulPartitionedCall2H
"upsample_8/StatefulPartitionedCall"upsample_8/StatefulPartitionedCall2H
"upsample_9/StatefulPartitionedCall"upsample_9/StatefulPartitionedCall:\ X
0
_output_shapes
:?????????`ΐ
$
_user_specified_name
imageInput


S__inference_batch_normalization_7_layer_call_and_return_conditional_losses_57382417

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ά
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+??????????????????????????? : : : : :*
epsilon%o:*
is_training( 2
FusedBatchNormV3
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+??????????????????????????? :::::i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
ͺ
¬
D__inference_conv_4_layer_call_and_return_conditional_losses_57378636

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@*
dtype02
Conv2D/ReadVariableOp€
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????2	
BiasAddm
IdentityIdentityBiasAdd:output:0*
T0*0
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????@:::W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs

°
S__inference_batch_normalization_3_layer_call_and_return_conditional_losses_57378671

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity’AssignNewValue’AssignNewValue_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype02
ReadVariableOp_1¨
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1έ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:?????????:::::*
epsilon%o:*
exponential_avg_factor%
Χ#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*0
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:?????????::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:X T
0
_output_shapes
:?????????
 
_user_specified_nameinputs
Ρ$
Ί
H__inference_upsample_6_layer_call_and_return_conditional_losses_57377685

inputs,
(conv2d_transpose_readvariableop_resource#
biasadd_readvariableop_resource
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2β
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2μ
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1x
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2μ
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_2P
mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
mul/y\
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: 2
mulP
add/yConst*
_output_shapes
: *
dtype0*
value	B : 2
add/yM
addAddV2mul:z:0add/y:output:0*
T0*
_output_shapes
: 2
addT
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2	
mul_1/yb
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: 2
mul_1T
add_1/yConst*
_output_shapes
: *
dtype0*
value	B : 2	
add_1/yU
add_1AddV2	mul_1:z:0add_1/y:output:0*
T0*
_output_shapes
: 2
add_1U
stack/3Const*
_output_shapes
: *
dtype0*
value
B :2	
stack/3
stackPackstrided_slice:output:0add:z:0	add_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:2
stackx
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2μ
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_3΅
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*(
_output_shapes
:*
dtype02!
conv2d_transpose/ReadVariableOpς
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*B
_output_shapes0
.:,???????????????????????????*
paddingVALID*
strides
2
conv2d_transpose
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp₯
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,???????????????????????????2	
BiasAdd
IdentityIdentityBiasAdd:output:0*
T0*B
_output_shapes0
.:,???????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:,???????????????????????????:::j f
B
_output_shapes0
.:,???????????????????????????
 
_user_specified_nameinputs
Τ

Q__inference_batch_normalization_layer_call_and_return_conditional_losses_57381343

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Λ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*L
_output_shapes:
8:?????????`ΐ:::::*
epsilon%o:*
is_training( 2
FusedBatchNormV3q
IdentityIdentityFusedBatchNormV3:y:0*
T0*0
_output_shapes
:?????????`ΐ2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:?????????`ΐ:::::X T
0
_output_shapes
:?????????`ΐ
 
_user_specified_nameinputs
§

S__inference_batch_normalization_3_layer_call_and_return_conditional_losses_57377520

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityu
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype02
ReadVariableOp_1¨
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1α
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,???????????????????????????:::::*
epsilon%o:*
is_training( 2
FusedBatchNormV3
IdentityIdentityFusedBatchNormV3:y:0*
T0*B
_output_shapes0
.:,???????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,???????????????????????????:::::j f
B
_output_shapes0
.:,???????????????????????????
 
_user_specified_nameinputs
ͺ
«
8__inference_batch_normalization_4_layer_call_fn_57381984

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity’StatefulPartitionedCall»
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,???????????????????????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *\
fWRU
S__inference_batch_normalization_4_layer_call_and_return_conditional_losses_573776052
StatefulPartitionedCall©
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,???????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,???????????????????????????::::22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,???????????????????????????
 
_user_specified_nameinputs

°
S__inference_batch_normalization_2_layer_call_and_return_conditional_losses_57381575

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity’AssignNewValue’AssignNewValue_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ψ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????0@:@:@:@:@:*
epsilon%o:*
exponential_avg_factor%
Χ#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*/
_output_shapes
:?????????0@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????0@::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:W S
/
_output_shapes
:?????????0@
 
_user_specified_nameinputs
ͺ
«
8__inference_batch_normalization_3_layer_call_fn_57381763

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity’StatefulPartitionedCall»
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,???????????????????????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *\
fWRU
S__inference_batch_normalization_3_layer_call_and_return_conditional_losses_573774892
StatefulPartitionedCall©
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,???????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,???????????????????????????::::22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,???????????????????????????
 
_user_specified_nameinputs
¨
«
8__inference_batch_normalization_2_layer_call_fn_57381683

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity’StatefulPartitionedCallΌ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *\
fWRU
S__inference_batch_normalization_2_layer_call_and_return_conditional_losses_573774042
StatefulPartitionedCall¨
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????@::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
ͺ
¬
D__inference_conv_1_layer_call_and_return_conditional_losses_57378297

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp€
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????`ΐ*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????`ΐ2	
BiasAddm
IdentityIdentityBiasAdd:output:0*
T0*0
_output_shapes
:?????????`ΐ2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:?????????`ΐ:::X T
0
_output_shapes
:?????????`ΐ
 
_user_specified_nameinputs

}
(__inference_final_layer_call_fn_57382706

inputs
unknown
	unknown_0
identity’StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????`ΐ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *L
fGRE
C__inference_final_layer_call_and_return_conditional_losses_573793932
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:?????????`ΐ2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:?????????`ΐ::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:?????????`ΐ
 
_user_specified_nameinputs
?

S__inference_batch_normalization_1_layer_call_and_return_conditional_losses_57378463

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Κ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????0` : : : : :*
epsilon%o:*
is_training( 2
FusedBatchNormV3p
IdentityIdentityFusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????0` 2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????0` :::::W S
/
_output_shapes
:?????????0` 
 
_user_specified_nameinputs
Χ
a
E__inference_re_lu_3_layer_call_and_return_conditional_losses_57381845

inputs
identityW
ReluReluinputs*
T0*0
_output_shapes
:?????????2
Reluo
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:?????????:X T
0
_output_shapes
:?????????
 
_user_specified_nameinputs

°
S__inference_batch_normalization_1_layer_call_and_return_conditional_losses_57381482

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity’AssignNewValue’AssignNewValue_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ψ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????0` : : : : :*
epsilon%o:*
exponential_avg_factor%
Χ#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*/
_output_shapes
:?????????0` 2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????0` ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:W S
/
_output_shapes
:?????????0` 
 
_user_specified_nameinputs
­
¬
D__inference_conv_6_layer_call_and_return_conditional_losses_57382030

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp€
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????2	
BiasAddm
IdentityIdentityBiasAdd:output:0*
T0*0
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:?????????:::X T
0
_output_shapes
:?????????
 
_user_specified_nameinputs
?

-__inference_upsample_8_layer_call_fn_57377999

inputs
unknown
	unknown_0
identity’StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *Q
fLRJ
H__inference_upsample_8_layer_call_and_return_conditional_losses_573779892
StatefulPartitionedCall¨
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+???????????????????????????@::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
¨
¬
D__inference_conv_7_layer_call_and_return_conditional_losses_57382200

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@*
dtype02
Conv2D/ReadVariableOp£
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????0@*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????0@2	
BiasAddl
IdentityIdentityBiasAdd:output:0*
T0*/
_output_shapes
:?????????0@2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:?????????0:::X T
0
_output_shapes
:?????????0
 
_user_specified_nameinputs
Λ
°
S__inference_batch_normalization_8_layer_call_and_return_conditional_losses_57378213

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity’AssignNewValue’AssignNewValue_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1κ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o:*
exponential_avg_factor%
Χ#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1¦
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
₯
¬
D__inference_conv_2_layer_call_and_return_conditional_losses_57381389

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOp£
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????0` *
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????0` 2	
BiasAddl
IdentityIdentityBiasAdd:output:0*
T0*/
_output_shapes
:?????????0` 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????0`:::W S
/
_output_shapes
:?????????0`
 
_user_specified_nameinputs
ή
«
8__inference_batch_normalization_1_layer_call_fn_57381513

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity’StatefulPartitionedCall¨
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????0` *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *\
fWRU
S__inference_batch_normalization_1_layer_call_and_return_conditional_losses_573784452
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????0` 2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????0` ::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????0` 
 
_user_specified_nameinputs
?

S__inference_batch_normalization_7_layer_call_and_return_conditional_losses_57382481

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Κ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????0` : : : : :*
epsilon%o:*
is_training( 2
FusedBatchNormV3p
IdentityIdentityFusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????0` 2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????0` :::::W S
/
_output_shapes
:?????????0` 
 
_user_specified_nameinputs

Φ
&__inference_signature_wrapper_57380334

imageinput
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

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28

unknown_29

unknown_30

unknown_31

unknown_32

unknown_33

unknown_34

unknown_35

unknown_36

unknown_37

unknown_38

unknown_39

unknown_40

unknown_41

unknown_42

unknown_43

unknown_44

unknown_45

unknown_46

unknown_47

unknown_48

unknown_49

unknown_50

unknown_51

unknown_52

unknown_53

unknown_54

unknown_55

unknown_56

unknown_57

unknown_58

unknown_59

unknown_60

unknown_61

unknown_62
identity’StatefulPartitionedCallΐ	
StatefulPartitionedCallStatefulPartitionedCall
imageinputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46
unknown_47
unknown_48
unknown_49
unknown_50
unknown_51
unknown_52
unknown_53
unknown_54
unknown_55
unknown_56
unknown_57
unknown_58
unknown_59
unknown_60
unknown_61
unknown_62*L
TinE
C2A*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????`ΐ*b
_read_only_resource_inputsD
B@	
 !"#$%&'()*+,-./0123456789:;<=>?@*2
config_proto" 

CPU

GPU2 *0J 8 *,
f'R%
#__inference__wrapped_model_573770792
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:?????????`ΐ2

Identity"
identityIdentity:output:0*±
_input_shapes
:?????????`ΐ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:\ X
0
_output_shapes
:?????????`ΐ
$
_user_specified_name
imageInput
Χ
a
E__inference_re_lu_3_layer_call_and_return_conditional_losses_57378730

inputs
identityW
ReluReluinputs*
T0*0
_output_shapes
:?????????2
Reluo
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:?????????:X T
0
_output_shapes
:?????????
 
_user_specified_nameinputs
Θ$
Ί
H__inference_upsample_9_layer_call_and_return_conditional_losses_57378141

inputs,
(conv2d_transpose_readvariableop_resource#
biasadd_readvariableop_resource
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2β
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2μ
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1x
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2μ
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_2P
mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
mul/y\
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: 2
mulP
add/yConst*
_output_shapes
: *
dtype0*
value	B : 2
add/yM
addAddV2mul:z:0add/y:output:0*
T0*
_output_shapes
: 2
addT
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2	
mul_1/yb
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: 2
mul_1T
add_1/yConst*
_output_shapes
: *
dtype0*
value	B : 2	
add_1/yU
add_1AddV2	mul_1:z:0add_1/y:output:0*
T0*
_output_shapes
: 2
add_1T
stack/3Const*
_output_shapes
: *
dtype0*
value	B :2	
stack/3
stackPackstrided_slice:output:0add:z:0	add_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:2
stackx
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2μ
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_3³
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype02!
conv2d_transpose/ReadVariableOpρ
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+???????????????????????????*
paddingVALID*
strides
2
conv2d_transpose
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp€
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????2	
BiasAdd~
IdentityIdentityBiasAdd:output:0*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+??????????????????????????? :::i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
ή
«
8__inference_batch_normalization_6_layer_call_fn_57382260

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity’StatefulPartitionedCall¨
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????0@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *\
fWRU
S__inference_batch_normalization_6_layer_call_and_return_conditional_losses_573790502
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????0@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????0@::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????0@
 
_user_specified_nameinputs
½
F
*__inference_re_lu_8_layer_call_fn_57382687

inputs
identityΡ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????`ΐ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_re_lu_8_layer_call_and_return_conditional_losses_573793752
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:?????????`ΐ2

Identity"
identityIdentity:output:0*/
_input_shapes
:?????????`ΐ:X T
0
_output_shapes
:?????????`ΐ
 
_user_specified_nameinputs
Ή
D
(__inference_re_lu_layer_call_fn_57381379

inputs
identityΟ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????`ΐ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *L
fGRE
C__inference_re_lu_layer_call_and_return_conditional_losses_573783912
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:?????????`ΐ2

Identity"
identityIdentity:output:0*/
_input_shapes
:?????????`ΐ:X T
0
_output_shapes
:?????????`ΐ
 
_user_specified_nameinputs
Λ
°
S__inference_batch_normalization_2_layer_call_and_return_conditional_losses_57381639

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity’AssignNewValue’AssignNewValue_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1κ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%o:*
exponential_avg_factor%
Χ#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1¦
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????@::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
Ά
N
2__inference_max_pooling2d_2_layer_call_fn_57377427

inputs
identityσ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *V
fQRO
M__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_573774212
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs

°
S__inference_batch_normalization_4_layer_call_and_return_conditional_losses_57378784

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity’AssignNewValue’AssignNewValue_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype02
ReadVariableOp_1¨
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1έ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:?????????:::::*
epsilon%o:*
exponential_avg_factor%
Χ#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*0
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:?????????::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:X T
0
_output_shapes
:?????????
 
_user_specified_nameinputs
Σ
a
E__inference_re_lu_7_layer_call_and_return_conditional_losses_57382512

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:?????????0` 2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:?????????0` 2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????0` :W S
/
_output_shapes
:?????????0` 
 
_user_specified_nameinputs
½
F
*__inference_re_lu_3_layer_call_fn_57381850

inputs
identityΡ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_re_lu_3_layer_call_and_return_conditional_losses_573787302
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:?????????:X T
0
_output_shapes
:?????????
 
_user_specified_nameinputs
Σ
a
E__inference_re_lu_6_layer_call_and_return_conditional_losses_57382342

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:?????????0@2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:?????????0@2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????0@:W S
/
_output_shapes
:?????????0@
 
_user_specified_nameinputs
Λ
°
S__inference_batch_normalization_6_layer_call_and_return_conditional_losses_57377909

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity’AssignNewValue’AssignNewValue_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1κ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%o:*
exponential_avg_factor%
Χ#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1¦
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????@::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs

~
)__inference_conv_3_layer_call_fn_57381555

inputs
unknown
	unknown_0
identity’StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????0@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *M
fHRF
D__inference_conv_3_layer_call_and_return_conditional_losses_573785232
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????0@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????0 ::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????0 
 
_user_specified_nameinputs

~
)__inference_conv_9_layer_call_fn_57382549

inputs
unknown
	unknown_0
identity’StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????`ΐ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *M
fHRF
D__inference_conv_9_layer_call_and_return_conditional_losses_573792812
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:?????????`ΐ2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:?????????`ΐ ::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:?????????`ΐ 
 
_user_specified_nameinputs
¦
«
8__inference_batch_normalization_6_layer_call_fn_57382324

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity’StatefulPartitionedCallΊ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *\
fWRU
S__inference_batch_normalization_6_layer_call_and_return_conditional_losses_573779092
StatefulPartitionedCall¨
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????@::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs


S__inference_batch_normalization_8_layer_call_and_return_conditional_losses_57382587

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ά
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o:*
is_training( 2
FusedBatchNormV3
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????:::::i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
Χ
a
E__inference_re_lu_4_layer_call_and_return_conditional_losses_57378843

inputs
identityW
ReluReluinputs*
T0*0
_output_shapes
:?????????2
Reluo
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:?????????:X T
0
_output_shapes
:?????????
 
_user_specified_nameinputs
τ΅
λ
J__inference_functional_1_layer_call_and_return_conditional_losses_57380934

inputs)
%conv_1_conv2d_readvariableop_resource*
&conv_1_biasadd_readvariableop_resource/
+batch_normalization_readvariableop_resource1
-batch_normalization_readvariableop_1_resource@
<batch_normalization_fusedbatchnormv3_readvariableop_resourceB
>batch_normalization_fusedbatchnormv3_readvariableop_1_resource)
%conv_2_conv2d_readvariableop_resource*
&conv_2_biasadd_readvariableop_resource1
-batch_normalization_1_readvariableop_resource3
/batch_normalization_1_readvariableop_1_resourceB
>batch_normalization_1_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource)
%conv_3_conv2d_readvariableop_resource*
&conv_3_biasadd_readvariableop_resource1
-batch_normalization_2_readvariableop_resource3
/batch_normalization_2_readvariableop_1_resourceB
>batch_normalization_2_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource)
%conv_4_conv2d_readvariableop_resource*
&conv_4_biasadd_readvariableop_resource1
-batch_normalization_3_readvariableop_resource3
/batch_normalization_3_readvariableop_1_resourceB
>batch_normalization_3_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource)
%conv_5_conv2d_readvariableop_resource*
&conv_5_biasadd_readvariableop_resource1
-batch_normalization_4_readvariableop_resource3
/batch_normalization_4_readvariableop_1_resourceB
>batch_normalization_4_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource7
3upsample_6_conv2d_transpose_readvariableop_resource.
*upsample_6_biasadd_readvariableop_resource)
%conv_6_conv2d_readvariableop_resource*
&conv_6_biasadd_readvariableop_resource1
-batch_normalization_5_readvariableop_resource3
/batch_normalization_5_readvariableop_1_resourceB
>batch_normalization_5_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource7
3upsample_7_conv2d_transpose_readvariableop_resource.
*upsample_7_biasadd_readvariableop_resource)
%conv_7_conv2d_readvariableop_resource*
&conv_7_biasadd_readvariableop_resource1
-batch_normalization_6_readvariableop_resource3
/batch_normalization_6_readvariableop_1_resourceB
>batch_normalization_6_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource7
3upsample_8_conv2d_transpose_readvariableop_resource.
*upsample_8_biasadd_readvariableop_resource)
%conv_8_conv2d_readvariableop_resource*
&conv_8_biasadd_readvariableop_resource1
-batch_normalization_7_readvariableop_resource3
/batch_normalization_7_readvariableop_1_resourceB
>batch_normalization_7_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource7
3upsample_9_conv2d_transpose_readvariableop_resource.
*upsample_9_biasadd_readvariableop_resource)
%conv_9_conv2d_readvariableop_resource*
&conv_9_biasadd_readvariableop_resource1
-batch_normalization_8_readvariableop_resource3
/batch_normalization_8_readvariableop_1_resourceB
>batch_normalization_8_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource(
$final_conv2d_readvariableop_resource)
%final_biasadd_readvariableop_resource
identity
tf_op_layer_RealDiv/RealDiv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?B2
tf_op_layer_RealDiv/RealDiv/yΏ
tf_op_layer_RealDiv/RealDivRealDivinputs&tf_op_layer_RealDiv/RealDiv/y:output:0*
T0*
_cloned(*0
_output_shapes
:?????????`ΐ2
tf_op_layer_RealDiv/RealDivs
tf_op_layer_Sub/Sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
tf_op_layer_Sub/Sub/yΌ
tf_op_layer_Sub/SubSubtf_op_layer_RealDiv/RealDiv:z:0tf_op_layer_Sub/Sub/y:output:0*
T0*
_cloned(*0
_output_shapes
:?????????`ΐ2
tf_op_layer_Sub/Subͺ
conv_1/Conv2D/ReadVariableOpReadVariableOp%conv_1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
conv_1/Conv2D/ReadVariableOpΚ
conv_1/Conv2DConv2Dtf_op_layer_Sub/Sub:z:0$conv_1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????`ΐ*
paddingSAME*
strides
2
conv_1/Conv2D‘
conv_1/BiasAdd/ReadVariableOpReadVariableOp&conv_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
conv_1/BiasAdd/ReadVariableOp₯
conv_1/BiasAddBiasAddconv_1/Conv2D:output:0%conv_1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????`ΐ2
conv_1/BiasAdd°
"batch_normalization/ReadVariableOpReadVariableOp+batch_normalization_readvariableop_resource*
_output_shapes
:*
dtype02$
"batch_normalization/ReadVariableOpΆ
$batch_normalization/ReadVariableOp_1ReadVariableOp-batch_normalization_readvariableop_1_resource*
_output_shapes
:*
dtype02&
$batch_normalization/ReadVariableOp_1γ
3batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOp<batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype025
3batch_normalization/FusedBatchNormV3/ReadVariableOpι
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp>batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype027
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1Τ
$batch_normalization/FusedBatchNormV3FusedBatchNormV3conv_1/BiasAdd:output:0*batch_normalization/ReadVariableOp:value:0,batch_normalization/ReadVariableOp_1:value:0;batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0=batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*L
_output_shapes:
8:?????????`ΐ:::::*
epsilon%o:*
is_training( 2&
$batch_normalization/FusedBatchNormV3

re_lu/ReluRelu(batch_normalization/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:?????????`ΐ2

re_lu/Reluΐ
max_pooling2d/MaxPoolMaxPoolre_lu/Relu:activations:0*/
_output_shapes
:?????????0`*
ksize
*
paddingVALID*
strides
2
max_pooling2d/MaxPoolͺ
conv_2/Conv2D/ReadVariableOpReadVariableOp%conv_2_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
conv_2/Conv2D/ReadVariableOpΠ
conv_2/Conv2DConv2Dmax_pooling2d/MaxPool:output:0$conv_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????0` *
paddingSAME*
strides
2
conv_2/Conv2D‘
conv_2/BiasAdd/ReadVariableOpReadVariableOp&conv_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02
conv_2/BiasAdd/ReadVariableOp€
conv_2/BiasAddBiasAddconv_2/Conv2D:output:0%conv_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????0` 2
conv_2/BiasAddΆ
$batch_normalization_1/ReadVariableOpReadVariableOp-batch_normalization_1_readvariableop_resource*
_output_shapes
: *
dtype02&
$batch_normalization_1/ReadVariableOpΌ
&batch_normalization_1/ReadVariableOp_1ReadVariableOp/batch_normalization_1_readvariableop_1_resource*
_output_shapes
: *
dtype02(
&batch_normalization_1/ReadVariableOp_1ι
5batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype027
5batch_normalization_1/FusedBatchNormV3/ReadVariableOpο
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype029
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ί
&batch_normalization_1/FusedBatchNormV3FusedBatchNormV3conv_2/BiasAdd:output:0,batch_normalization_1/ReadVariableOp:value:0.batch_normalization_1/ReadVariableOp_1:value:0=batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????0` : : : : :*
epsilon%o:*
is_training( 2(
&batch_normalization_1/FusedBatchNormV3
re_lu_1/ReluRelu*batch_normalization_1/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????0` 2
re_lu_1/ReluΖ
max_pooling2d_1/MaxPoolMaxPoolre_lu_1/Relu:activations:0*/
_output_shapes
:?????????0 *
ksize
*
paddingVALID*
strides
2
max_pooling2d_1/MaxPoolͺ
conv_3/Conv2D/ReadVariableOpReadVariableOp%conv_3_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02
conv_3/Conv2D/ReadVariableOp?
conv_3/Conv2DConv2D max_pooling2d_1/MaxPool:output:0$conv_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????0@*
paddingSAME*
strides
2
conv_3/Conv2D‘
conv_3/BiasAdd/ReadVariableOpReadVariableOp&conv_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
conv_3/BiasAdd/ReadVariableOp€
conv_3/BiasAddBiasAddconv_3/Conv2D:output:0%conv_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????0@2
conv_3/BiasAddΆ
$batch_normalization_2/ReadVariableOpReadVariableOp-batch_normalization_2_readvariableop_resource*
_output_shapes
:@*
dtype02&
$batch_normalization_2/ReadVariableOpΌ
&batch_normalization_2/ReadVariableOp_1ReadVariableOp/batch_normalization_2_readvariableop_1_resource*
_output_shapes
:@*
dtype02(
&batch_normalization_2/ReadVariableOp_1ι
5batch_normalization_2/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype027
5batch_normalization_2/FusedBatchNormV3/ReadVariableOpο
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype029
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1ί
&batch_normalization_2/FusedBatchNormV3FusedBatchNormV3conv_3/BiasAdd:output:0,batch_normalization_2/ReadVariableOp:value:0.batch_normalization_2/ReadVariableOp_1:value:0=batch_normalization_2/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????0@:@:@:@:@:*
epsilon%o:*
is_training( 2(
&batch_normalization_2/FusedBatchNormV3
re_lu_2/ReluRelu*batch_normalization_2/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????0@2
re_lu_2/ReluΖ
max_pooling2d_2/MaxPoolMaxPoolre_lu_2/Relu:activations:0*/
_output_shapes
:?????????@*
ksize
*
paddingVALID*
strides
2
max_pooling2d_2/MaxPool«
conv_4/Conv2D/ReadVariableOpReadVariableOp%conv_4_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype02
conv_4/Conv2D/ReadVariableOpΣ
conv_4/Conv2DConv2D max_pooling2d_2/MaxPool:output:0$conv_4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????*
paddingSAME*
strides
2
conv_4/Conv2D’
conv_4/BiasAdd/ReadVariableOpReadVariableOp&conv_4_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
conv_4/BiasAdd/ReadVariableOp₯
conv_4/BiasAddBiasAddconv_4/Conv2D:output:0%conv_4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????2
conv_4/BiasAdd·
$batch_normalization_3/ReadVariableOpReadVariableOp-batch_normalization_3_readvariableop_resource*
_output_shapes	
:*
dtype02&
$batch_normalization_3/ReadVariableOp½
&batch_normalization_3/ReadVariableOp_1ReadVariableOp/batch_normalization_3_readvariableop_1_resource*
_output_shapes	
:*
dtype02(
&batch_normalization_3/ReadVariableOp_1κ
5batch_normalization_3/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_3_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype027
5batch_normalization_3/FusedBatchNormV3/ReadVariableOpπ
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype029
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1δ
&batch_normalization_3/FusedBatchNormV3FusedBatchNormV3conv_4/BiasAdd:output:0,batch_normalization_3/ReadVariableOp:value:0.batch_normalization_3/ReadVariableOp_1:value:0=batch_normalization_3/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:?????????:::::*
epsilon%o:*
is_training( 2(
&batch_normalization_3/FusedBatchNormV3
re_lu_3/ReluRelu*batch_normalization_3/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:?????????2
re_lu_3/ReluΗ
max_pooling2d_3/MaxPoolMaxPoolre_lu_3/Relu:activations:0*0
_output_shapes
:?????????*
ksize
*
paddingVALID*
strides
2
max_pooling2d_3/MaxPool¬
conv_5/Conv2D/ReadVariableOpReadVariableOp%conv_5_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02
conv_5/Conv2D/ReadVariableOpΣ
conv_5/Conv2DConv2D max_pooling2d_3/MaxPool:output:0$conv_5/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????*
paddingSAME*
strides
2
conv_5/Conv2D’
conv_5/BiasAdd/ReadVariableOpReadVariableOp&conv_5_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
conv_5/BiasAdd/ReadVariableOp₯
conv_5/BiasAddBiasAddconv_5/Conv2D:output:0%conv_5/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????2
conv_5/BiasAdd·
$batch_normalization_4/ReadVariableOpReadVariableOp-batch_normalization_4_readvariableop_resource*
_output_shapes	
:*
dtype02&
$batch_normalization_4/ReadVariableOp½
&batch_normalization_4/ReadVariableOp_1ReadVariableOp/batch_normalization_4_readvariableop_1_resource*
_output_shapes	
:*
dtype02(
&batch_normalization_4/ReadVariableOp_1κ
5batch_normalization_4/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_4_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype027
5batch_normalization_4/FusedBatchNormV3/ReadVariableOpπ
7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype029
7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1δ
&batch_normalization_4/FusedBatchNormV3FusedBatchNormV3conv_5/BiasAdd:output:0,batch_normalization_4/ReadVariableOp:value:0.batch_normalization_4/ReadVariableOp_1:value:0=batch_normalization_4/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:?????????:::::*
epsilon%o:*
is_training( 2(
&batch_normalization_4/FusedBatchNormV3
re_lu_4/ReluRelu*batch_normalization_4/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:?????????2
re_lu_4/Relun
upsample_6/ShapeShapere_lu_4/Relu:activations:0*
T0*
_output_shapes
:2
upsample_6/Shape
upsample_6/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
upsample_6/strided_slice/stack
 upsample_6/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 upsample_6/strided_slice/stack_1
 upsample_6/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 upsample_6/strided_slice/stack_2€
upsample_6/strided_sliceStridedSliceupsample_6/Shape:output:0'upsample_6/strided_slice/stack:output:0)upsample_6/strided_slice/stack_1:output:0)upsample_6/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
upsample_6/strided_slicej
upsample_6/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2
upsample_6/stack/1j
upsample_6/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2
upsample_6/stack/2k
upsample_6/stack/3Const*
_output_shapes
: *
dtype0*
value
B :2
upsample_6/stack/3Τ
upsample_6/stackPack!upsample_6/strided_slice:output:0upsample_6/stack/1:output:0upsample_6/stack/2:output:0upsample_6/stack/3:output:0*
N*
T0*
_output_shapes
:2
upsample_6/stack
 upsample_6/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2"
 upsample_6/strided_slice_1/stack
"upsample_6/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2$
"upsample_6/strided_slice_1/stack_1
"upsample_6/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"upsample_6/strided_slice_1/stack_2?
upsample_6/strided_slice_1StridedSliceupsample_6/stack:output:0)upsample_6/strided_slice_1/stack:output:0+upsample_6/strided_slice_1/stack_1:output:0+upsample_6/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
upsample_6/strided_slice_1Φ
*upsample_6/conv2d_transpose/ReadVariableOpReadVariableOp3upsample_6_conv2d_transpose_readvariableop_resource*(
_output_shapes
:*
dtype02,
*upsample_6/conv2d_transpose/ReadVariableOp 
upsample_6/conv2d_transposeConv2DBackpropInputupsample_6/stack:output:02upsample_6/conv2d_transpose/ReadVariableOp:value:0re_lu_4/Relu:activations:0*
T0*0
_output_shapes
:?????????*
paddingVALID*
strides
2
upsample_6/conv2d_transpose?
!upsample_6/BiasAdd/ReadVariableOpReadVariableOp*upsample_6_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02#
!upsample_6/BiasAdd/ReadVariableOpΏ
upsample_6/BiasAddBiasAdd$upsample_6/conv2d_transpose:output:0)upsample_6/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????2
upsample_6/BiasAdd
"tf_op_layer_concat_6/concat_6/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2$
"tf_op_layer_concat_6/concat_6/axis
tf_op_layer_concat_6/concat_6ConcatV2upsample_6/BiasAdd:output:0re_lu_3/Relu:activations:0+tf_op_layer_concat_6/concat_6/axis:output:0*
N*
T0*
_cloned(*0
_output_shapes
:?????????2
tf_op_layer_concat_6/concat_6¬
conv_6/Conv2D/ReadVariableOpReadVariableOp%conv_6_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02
conv_6/Conv2D/ReadVariableOpΩ
conv_6/Conv2DConv2D&tf_op_layer_concat_6/concat_6:output:0$conv_6/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????*
paddingSAME*
strides
2
conv_6/Conv2D’
conv_6/BiasAdd/ReadVariableOpReadVariableOp&conv_6_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
conv_6/BiasAdd/ReadVariableOp₯
conv_6/BiasAddBiasAddconv_6/Conv2D:output:0%conv_6/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????2
conv_6/BiasAdd·
$batch_normalization_5/ReadVariableOpReadVariableOp-batch_normalization_5_readvariableop_resource*
_output_shapes	
:*
dtype02&
$batch_normalization_5/ReadVariableOp½
&batch_normalization_5/ReadVariableOp_1ReadVariableOp/batch_normalization_5_readvariableop_1_resource*
_output_shapes	
:*
dtype02(
&batch_normalization_5/ReadVariableOp_1κ
5batch_normalization_5/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_5_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype027
5batch_normalization_5/FusedBatchNormV3/ReadVariableOpπ
7batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype029
7batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1δ
&batch_normalization_5/FusedBatchNormV3FusedBatchNormV3conv_6/BiasAdd:output:0,batch_normalization_5/ReadVariableOp:value:0.batch_normalization_5/ReadVariableOp_1:value:0=batch_normalization_5/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:?????????:::::*
epsilon%o:*
is_training( 2(
&batch_normalization_5/FusedBatchNormV3
re_lu_5/ReluRelu*batch_normalization_5/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:?????????2
re_lu_5/Relun
upsample_7/ShapeShapere_lu_5/Relu:activations:0*
T0*
_output_shapes
:2
upsample_7/Shape
upsample_7/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
upsample_7/strided_slice/stack
 upsample_7/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 upsample_7/strided_slice/stack_1
 upsample_7/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 upsample_7/strided_slice/stack_2€
upsample_7/strided_sliceStridedSliceupsample_7/Shape:output:0'upsample_7/strided_slice/stack:output:0)upsample_7/strided_slice/stack_1:output:0)upsample_7/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
upsample_7/strided_slicej
upsample_7/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2
upsample_7/stack/1j
upsample_7/stack/2Const*
_output_shapes
: *
dtype0*
value	B :02
upsample_7/stack/2j
upsample_7/stack/3Const*
_output_shapes
: *
dtype0*
value	B :@2
upsample_7/stack/3Τ
upsample_7/stackPack!upsample_7/strided_slice:output:0upsample_7/stack/1:output:0upsample_7/stack/2:output:0upsample_7/stack/3:output:0*
N*
T0*
_output_shapes
:2
upsample_7/stack
 upsample_7/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2"
 upsample_7/strided_slice_1/stack
"upsample_7/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2$
"upsample_7/strided_slice_1/stack_1
"upsample_7/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"upsample_7/strided_slice_1/stack_2?
upsample_7/strided_slice_1StridedSliceupsample_7/stack:output:0)upsample_7/strided_slice_1/stack:output:0+upsample_7/strided_slice_1/stack_1:output:0+upsample_7/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
upsample_7/strided_slice_1Υ
*upsample_7/conv2d_transpose/ReadVariableOpReadVariableOp3upsample_7_conv2d_transpose_readvariableop_resource*'
_output_shapes
:@*
dtype02,
*upsample_7/conv2d_transpose/ReadVariableOp
upsample_7/conv2d_transposeConv2DBackpropInputupsample_7/stack:output:02upsample_7/conv2d_transpose/ReadVariableOp:value:0re_lu_5/Relu:activations:0*
T0*/
_output_shapes
:?????????0@*
paddingVALID*
strides
2
upsample_7/conv2d_transpose­
!upsample_7/BiasAdd/ReadVariableOpReadVariableOp*upsample_7_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02#
!upsample_7/BiasAdd/ReadVariableOpΎ
upsample_7/BiasAddBiasAdd$upsample_7/conv2d_transpose:output:0)upsample_7/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????0@2
upsample_7/BiasAdd
"tf_op_layer_concat_7/concat_7/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2$
"tf_op_layer_concat_7/concat_7/axis
tf_op_layer_concat_7/concat_7ConcatV2upsample_7/BiasAdd:output:0re_lu_2/Relu:activations:0+tf_op_layer_concat_7/concat_7/axis:output:0*
N*
T0*
_cloned(*0
_output_shapes
:?????????02
tf_op_layer_concat_7/concat_7«
conv_7/Conv2D/ReadVariableOpReadVariableOp%conv_7_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype02
conv_7/Conv2D/ReadVariableOpΨ
conv_7/Conv2DConv2D&tf_op_layer_concat_7/concat_7:output:0$conv_7/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????0@*
paddingSAME*
strides
2
conv_7/Conv2D‘
conv_7/BiasAdd/ReadVariableOpReadVariableOp&conv_7_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
conv_7/BiasAdd/ReadVariableOp€
conv_7/BiasAddBiasAddconv_7/Conv2D:output:0%conv_7/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????0@2
conv_7/BiasAddΆ
$batch_normalization_6/ReadVariableOpReadVariableOp-batch_normalization_6_readvariableop_resource*
_output_shapes
:@*
dtype02&
$batch_normalization_6/ReadVariableOpΌ
&batch_normalization_6/ReadVariableOp_1ReadVariableOp/batch_normalization_6_readvariableop_1_resource*
_output_shapes
:@*
dtype02(
&batch_normalization_6/ReadVariableOp_1ι
5batch_normalization_6/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_6_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype027
5batch_normalization_6/FusedBatchNormV3/ReadVariableOpο
7batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype029
7batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1ί
&batch_normalization_6/FusedBatchNormV3FusedBatchNormV3conv_7/BiasAdd:output:0,batch_normalization_6/ReadVariableOp:value:0.batch_normalization_6/ReadVariableOp_1:value:0=batch_normalization_6/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????0@:@:@:@:@:*
epsilon%o:*
is_training( 2(
&batch_normalization_6/FusedBatchNormV3
re_lu_6/ReluRelu*batch_normalization_6/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????0@2
re_lu_6/Relun
upsample_8/ShapeShapere_lu_6/Relu:activations:0*
T0*
_output_shapes
:2
upsample_8/Shape
upsample_8/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
upsample_8/strided_slice/stack
 upsample_8/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 upsample_8/strided_slice/stack_1
 upsample_8/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 upsample_8/strided_slice/stack_2€
upsample_8/strided_sliceStridedSliceupsample_8/Shape:output:0'upsample_8/strided_slice/stack:output:0)upsample_8/strided_slice/stack_1:output:0)upsample_8/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
upsample_8/strided_slicej
upsample_8/stack/1Const*
_output_shapes
: *
dtype0*
value	B :02
upsample_8/stack/1j
upsample_8/stack/2Const*
_output_shapes
: *
dtype0*
value	B :`2
upsample_8/stack/2j
upsample_8/stack/3Const*
_output_shapes
: *
dtype0*
value	B : 2
upsample_8/stack/3Τ
upsample_8/stackPack!upsample_8/strided_slice:output:0upsample_8/stack/1:output:0upsample_8/stack/2:output:0upsample_8/stack/3:output:0*
N*
T0*
_output_shapes
:2
upsample_8/stack
 upsample_8/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2"
 upsample_8/strided_slice_1/stack
"upsample_8/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2$
"upsample_8/strided_slice_1/stack_1
"upsample_8/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"upsample_8/strided_slice_1/stack_2?
upsample_8/strided_slice_1StridedSliceupsample_8/stack:output:0)upsample_8/strided_slice_1/stack:output:0+upsample_8/strided_slice_1/stack_1:output:0+upsample_8/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
upsample_8/strided_slice_1Τ
*upsample_8/conv2d_transpose/ReadVariableOpReadVariableOp3upsample_8_conv2d_transpose_readvariableop_resource*&
_output_shapes
: @*
dtype02,
*upsample_8/conv2d_transpose/ReadVariableOp
upsample_8/conv2d_transposeConv2DBackpropInputupsample_8/stack:output:02upsample_8/conv2d_transpose/ReadVariableOp:value:0re_lu_6/Relu:activations:0*
T0*/
_output_shapes
:?????????0` *
paddingVALID*
strides
2
upsample_8/conv2d_transpose­
!upsample_8/BiasAdd/ReadVariableOpReadVariableOp*upsample_8_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02#
!upsample_8/BiasAdd/ReadVariableOpΎ
upsample_8/BiasAddBiasAdd$upsample_8/conv2d_transpose:output:0)upsample_8/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????0` 2
upsample_8/BiasAdd
"tf_op_layer_concat_8/concat_8/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2$
"tf_op_layer_concat_8/concat_8/axis
tf_op_layer_concat_8/concat_8ConcatV2upsample_8/BiasAdd:output:0re_lu_1/Relu:activations:0+tf_op_layer_concat_8/concat_8/axis:output:0*
N*
T0*
_cloned(*/
_output_shapes
:?????????0`@2
tf_op_layer_concat_8/concat_8ͺ
conv_8/Conv2D/ReadVariableOpReadVariableOp%conv_8_conv2d_readvariableop_resource*&
_output_shapes
:@ *
dtype02
conv_8/Conv2D/ReadVariableOpΨ
conv_8/Conv2DConv2D&tf_op_layer_concat_8/concat_8:output:0$conv_8/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????0` *
paddingSAME*
strides
2
conv_8/Conv2D‘
conv_8/BiasAdd/ReadVariableOpReadVariableOp&conv_8_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02
conv_8/BiasAdd/ReadVariableOp€
conv_8/BiasAddBiasAddconv_8/Conv2D:output:0%conv_8/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????0` 2
conv_8/BiasAddΆ
$batch_normalization_7/ReadVariableOpReadVariableOp-batch_normalization_7_readvariableop_resource*
_output_shapes
: *
dtype02&
$batch_normalization_7/ReadVariableOpΌ
&batch_normalization_7/ReadVariableOp_1ReadVariableOp/batch_normalization_7_readvariableop_1_resource*
_output_shapes
: *
dtype02(
&batch_normalization_7/ReadVariableOp_1ι
5batch_normalization_7/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_7_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype027
5batch_normalization_7/FusedBatchNormV3/ReadVariableOpο
7batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype029
7batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1ί
&batch_normalization_7/FusedBatchNormV3FusedBatchNormV3conv_8/BiasAdd:output:0,batch_normalization_7/ReadVariableOp:value:0.batch_normalization_7/ReadVariableOp_1:value:0=batch_normalization_7/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????0` : : : : :*
epsilon%o:*
is_training( 2(
&batch_normalization_7/FusedBatchNormV3
re_lu_7/ReluRelu*batch_normalization_7/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????0` 2
re_lu_7/Relun
upsample_9/ShapeShapere_lu_7/Relu:activations:0*
T0*
_output_shapes
:2
upsample_9/Shape
upsample_9/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
upsample_9/strided_slice/stack
 upsample_9/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 upsample_9/strided_slice/stack_1
 upsample_9/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 upsample_9/strided_slice/stack_2€
upsample_9/strided_sliceStridedSliceupsample_9/Shape:output:0'upsample_9/strided_slice/stack:output:0)upsample_9/strided_slice/stack_1:output:0)upsample_9/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
upsample_9/strided_slicej
upsample_9/stack/1Const*
_output_shapes
: *
dtype0*
value	B :`2
upsample_9/stack/1k
upsample_9/stack/2Const*
_output_shapes
: *
dtype0*
value
B :ΐ2
upsample_9/stack/2j
upsample_9/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2
upsample_9/stack/3Τ
upsample_9/stackPack!upsample_9/strided_slice:output:0upsample_9/stack/1:output:0upsample_9/stack/2:output:0upsample_9/stack/3:output:0*
N*
T0*
_output_shapes
:2
upsample_9/stack
 upsample_9/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2"
 upsample_9/strided_slice_1/stack
"upsample_9/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2$
"upsample_9/strided_slice_1/stack_1
"upsample_9/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"upsample_9/strided_slice_1/stack_2?
upsample_9/strided_slice_1StridedSliceupsample_9/stack:output:0)upsample_9/strided_slice_1/stack:output:0+upsample_9/strided_slice_1/stack_1:output:0+upsample_9/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
upsample_9/strided_slice_1Τ
*upsample_9/conv2d_transpose/ReadVariableOpReadVariableOp3upsample_9_conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype02,
*upsample_9/conv2d_transpose/ReadVariableOp 
upsample_9/conv2d_transposeConv2DBackpropInputupsample_9/stack:output:02upsample_9/conv2d_transpose/ReadVariableOp:value:0re_lu_7/Relu:activations:0*
T0*0
_output_shapes
:?????????`ΐ*
paddingVALID*
strides
2
upsample_9/conv2d_transpose­
!upsample_9/BiasAdd/ReadVariableOpReadVariableOp*upsample_9_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02#
!upsample_9/BiasAdd/ReadVariableOpΏ
upsample_9/BiasAddBiasAdd$upsample_9/conv2d_transpose:output:0)upsample_9/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????`ΐ2
upsample_9/BiasAdd
"tf_op_layer_concat_9/concat_9/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2$
"tf_op_layer_concat_9/concat_9/axis
tf_op_layer_concat_9/concat_9ConcatV2upsample_9/BiasAdd:output:0re_lu/Relu:activations:0+tf_op_layer_concat_9/concat_9/axis:output:0*
N*
T0*
_cloned(*0
_output_shapes
:?????????`ΐ 2
tf_op_layer_concat_9/concat_9ͺ
conv_9/Conv2D/ReadVariableOpReadVariableOp%conv_9_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
conv_9/Conv2D/ReadVariableOpΩ
conv_9/Conv2DConv2D&tf_op_layer_concat_9/concat_9:output:0$conv_9/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????`ΐ*
paddingSAME*
strides
2
conv_9/Conv2D‘
conv_9/BiasAdd/ReadVariableOpReadVariableOp&conv_9_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
conv_9/BiasAdd/ReadVariableOp₯
conv_9/BiasAddBiasAddconv_9/Conv2D:output:0%conv_9/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????`ΐ2
conv_9/BiasAddΆ
$batch_normalization_8/ReadVariableOpReadVariableOp-batch_normalization_8_readvariableop_resource*
_output_shapes
:*
dtype02&
$batch_normalization_8/ReadVariableOpΌ
&batch_normalization_8/ReadVariableOp_1ReadVariableOp/batch_normalization_8_readvariableop_1_resource*
_output_shapes
:*
dtype02(
&batch_normalization_8/ReadVariableOp_1ι
5batch_normalization_8/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_8_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype027
5batch_normalization_8/FusedBatchNormV3/ReadVariableOpο
7batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype029
7batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1ΰ
&batch_normalization_8/FusedBatchNormV3FusedBatchNormV3conv_9/BiasAdd:output:0,batch_normalization_8/ReadVariableOp:value:0.batch_normalization_8/ReadVariableOp_1:value:0=batch_normalization_8/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*L
_output_shapes:
8:?????????`ΐ:::::*
epsilon%o:*
is_training( 2(
&batch_normalization_8/FusedBatchNormV3
re_lu_8/ReluRelu*batch_normalization_8/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:?????????`ΐ2
re_lu_8/Relu§
final/Conv2D/ReadVariableOpReadVariableOp$final_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
final/Conv2D/ReadVariableOpΚ
final/Conv2DConv2Dre_lu_8/Relu:activations:0#final/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????`ΐ*
paddingSAME*
strides
2
final/Conv2D
final/BiasAdd/ReadVariableOpReadVariableOp%final_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
final/BiasAdd/ReadVariableOp‘
final/BiasAddBiasAddfinal/Conv2D:output:0$final/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????`ΐ2
final/BiasAdds
IdentityIdentityfinal/BiasAdd:output:0*
T0*0
_output_shapes
:?????????`ΐ2

Identity"
identityIdentity:output:0*±
_input_shapes
:?????????`ΐ:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::X T
0
_output_shapes
:?????????`ΐ
 
_user_specified_nameinputs
§

S__inference_batch_normalization_3_layer_call_and_return_conditional_losses_57381750

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityu
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype02
ReadVariableOp_1¨
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1α
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,???????????????????????????:::::*
epsilon%o:*
is_training( 2
FusedBatchNormV3
IdentityIdentityFusedBatchNormV3:y:0*
T0*B
_output_shapes0
.:,???????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,???????????????????????????:::::j f
B
_output_shapes0
.:,???????????????????????????
 
_user_specified_nameinputs
δ
«
8__inference_batch_normalization_4_layer_call_fn_57381933

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity’StatefulPartitionedCall«
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *\
fWRU
S__inference_batch_normalization_4_layer_call_and_return_conditional_losses_573788022
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:?????????::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:?????????
 
_user_specified_nameinputs
Ι
?
Q__inference_batch_normalization_layer_call_and_return_conditional_losses_57381261

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity’AssignNewValue’AssignNewValue_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1κ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o:*
exponential_avg_factor%
Χ#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1¦
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
ͺ
¬
D__inference_conv_9_layer_call_and_return_conditional_losses_57382540

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOp€
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????`ΐ*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????`ΐ2	
BiasAddm
IdentityIdentityBiasAdd:output:0*
T0*0
_output_shapes
:?????????`ΐ2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:?????????`ΐ :::X T
0
_output_shapes
:?????????`ΐ 
 
_user_specified_nameinputs
Ώ
~
R__inference_tf_op_layer_concat_8_layer_call_and_return_conditional_losses_57382354
inputs_0
inputs_1
identityi
concat_8/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
concat_8/axis
concat_8ConcatV2inputs_0inputs_1concat_8/axis:output:0*
N*
T0*
_cloned(*/
_output_shapes
:?????????0`@2

concat_8m
IdentityIdentityconcat_8:output:0*
T0*/
_output_shapes
:?????????0`@2

Identity"
identityIdentity:output:0*[
_input_shapesJ
H:+??????????????????????????? :?????????0` :k g
A
_output_shapes/
-:+??????????????????????????? 
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:?????????0` 
"
_user_specified_name
inputs/1
ϊΙ
ζ
J__inference_functional_1_layer_call_and_return_conditional_losses_57379761

inputs
conv_1_57379592
conv_1_57379594 
batch_normalization_57379597 
batch_normalization_57379599 
batch_normalization_57379601 
batch_normalization_57379603
conv_2_57379608
conv_2_57379610"
batch_normalization_1_57379613"
batch_normalization_1_57379615"
batch_normalization_1_57379617"
batch_normalization_1_57379619
conv_3_57379624
conv_3_57379626"
batch_normalization_2_57379629"
batch_normalization_2_57379631"
batch_normalization_2_57379633"
batch_normalization_2_57379635
conv_4_57379640
conv_4_57379642"
batch_normalization_3_57379645"
batch_normalization_3_57379647"
batch_normalization_3_57379649"
batch_normalization_3_57379651
conv_5_57379656
conv_5_57379658"
batch_normalization_4_57379661"
batch_normalization_4_57379663"
batch_normalization_4_57379665"
batch_normalization_4_57379667
upsample_6_57379671
upsample_6_57379673
conv_6_57379677
conv_6_57379679"
batch_normalization_5_57379682"
batch_normalization_5_57379684"
batch_normalization_5_57379686"
batch_normalization_5_57379688
upsample_7_57379692
upsample_7_57379694
conv_7_57379698
conv_7_57379700"
batch_normalization_6_57379703"
batch_normalization_6_57379705"
batch_normalization_6_57379707"
batch_normalization_6_57379709
upsample_8_57379713
upsample_8_57379715
conv_8_57379719
conv_8_57379721"
batch_normalization_7_57379724"
batch_normalization_7_57379726"
batch_normalization_7_57379728"
batch_normalization_7_57379730
upsample_9_57379734
upsample_9_57379736
conv_9_57379740
conv_9_57379742"
batch_normalization_8_57379745"
batch_normalization_8_57379747"
batch_normalization_8_57379749"
batch_normalization_8_57379751
final_57379755
final_57379757
identity’+batch_normalization/StatefulPartitionedCall’-batch_normalization_1/StatefulPartitionedCall’-batch_normalization_2/StatefulPartitionedCall’-batch_normalization_3/StatefulPartitionedCall’-batch_normalization_4/StatefulPartitionedCall’-batch_normalization_5/StatefulPartitionedCall’-batch_normalization_6/StatefulPartitionedCall’-batch_normalization_7/StatefulPartitionedCall’-batch_normalization_8/StatefulPartitionedCall’conv_1/StatefulPartitionedCall’conv_2/StatefulPartitionedCall’conv_3/StatefulPartitionedCall’conv_4/StatefulPartitionedCall’conv_5/StatefulPartitionedCall’conv_6/StatefulPartitionedCall’conv_7/StatefulPartitionedCall’conv_8/StatefulPartitionedCall’conv_9/StatefulPartitionedCall’final/StatefulPartitionedCall’"upsample_6/StatefulPartitionedCall’"upsample_7/StatefulPartitionedCall’"upsample_8/StatefulPartitionedCall’"upsample_9/StatefulPartitionedCall
#tf_op_layer_RealDiv/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????`ΐ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *Z
fURS
Q__inference_tf_op_layer_RealDiv_layer_call_and_return_conditional_losses_573782652%
#tf_op_layer_RealDiv/PartitionedCall
tf_op_layer_Sub/PartitionedCallPartitionedCall,tf_op_layer_RealDiv/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????`ΐ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *V
fQRO
M__inference_tf_op_layer_Sub_layer_call_and_return_conditional_losses_573782792!
tf_op_layer_Sub/PartitionedCallΐ
conv_1/StatefulPartitionedCallStatefulPartitionedCall(tf_op_layer_Sub/PartitionedCall:output:0conv_1_57379592conv_1_57379594*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????`ΐ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *M
fHRF
D__inference_conv_1_layer_call_and_return_conditional_losses_573782972 
conv_1/StatefulPartitionedCallΎ
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall'conv_1/StatefulPartitionedCall:output:0batch_normalization_57379597batch_normalization_57379599batch_normalization_57379601batch_normalization_57379603*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????`ΐ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *Z
fURS
Q__inference_batch_normalization_layer_call_and_return_conditional_losses_573783322-
+batch_normalization/StatefulPartitionedCall
re_lu/PartitionedCallPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????`ΐ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *L
fGRE
C__inference_re_lu_layer_call_and_return_conditional_losses_573783912
re_lu/PartitionedCall
max_pooling2d/PartitionedCallPartitionedCallre_lu/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????0`* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *T
fORM
K__inference_max_pooling2d_layer_call_and_return_conditional_losses_573771892
max_pooling2d/PartitionedCall½
conv_2/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0conv_2_57379608conv_2_57379610*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????0` *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *M
fHRF
D__inference_conv_2_layer_call_and_return_conditional_losses_573784102 
conv_2/StatefulPartitionedCallΛ
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall'conv_2/StatefulPartitionedCall:output:0batch_normalization_1_57379613batch_normalization_1_57379615batch_normalization_1_57379617batch_normalization_1_57379619*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????0` *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *\
fWRU
S__inference_batch_normalization_1_layer_call_and_return_conditional_losses_573784452/
-batch_normalization_1/StatefulPartitionedCall
re_lu_1/PartitionedCallPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????0` * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_re_lu_1_layer_call_and_return_conditional_losses_573785042
re_lu_1/PartitionedCall
max_pooling2d_1/PartitionedCallPartitionedCall re_lu_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????0 * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *V
fQRO
M__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_573773052!
max_pooling2d_1/PartitionedCallΏ
conv_3/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_1/PartitionedCall:output:0conv_3_57379624conv_3_57379626*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????0@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *M
fHRF
D__inference_conv_3_layer_call_and_return_conditional_losses_573785232 
conv_3/StatefulPartitionedCallΛ
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall'conv_3/StatefulPartitionedCall:output:0batch_normalization_2_57379629batch_normalization_2_57379631batch_normalization_2_57379633batch_normalization_2_57379635*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????0@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *\
fWRU
S__inference_batch_normalization_2_layer_call_and_return_conditional_losses_573785582/
-batch_normalization_2/StatefulPartitionedCall
re_lu_2/PartitionedCallPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????0@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_re_lu_2_layer_call_and_return_conditional_losses_573786172
re_lu_2/PartitionedCall
max_pooling2d_2/PartitionedCallPartitionedCall re_lu_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *V
fQRO
M__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_573774212!
max_pooling2d_2/PartitionedCallΐ
conv_4/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_2/PartitionedCall:output:0conv_4_57379640conv_4_57379642*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *M
fHRF
D__inference_conv_4_layer_call_and_return_conditional_losses_573786362 
conv_4/StatefulPartitionedCallΜ
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCall'conv_4/StatefulPartitionedCall:output:0batch_normalization_3_57379645batch_normalization_3_57379647batch_normalization_3_57379649batch_normalization_3_57379651*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *\
fWRU
S__inference_batch_normalization_3_layer_call_and_return_conditional_losses_573786712/
-batch_normalization_3/StatefulPartitionedCall
re_lu_3/PartitionedCallPartitionedCall6batch_normalization_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_re_lu_3_layer_call_and_return_conditional_losses_573787302
re_lu_3/PartitionedCall
max_pooling2d_3/PartitionedCallPartitionedCall re_lu_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *V
fQRO
M__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_573775372!
max_pooling2d_3/PartitionedCallΐ
conv_5/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_3/PartitionedCall:output:0conv_5_57379656conv_5_57379658*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *M
fHRF
D__inference_conv_5_layer_call_and_return_conditional_losses_573787492 
conv_5/StatefulPartitionedCallΜ
-batch_normalization_4/StatefulPartitionedCallStatefulPartitionedCall'conv_5/StatefulPartitionedCall:output:0batch_normalization_4_57379661batch_normalization_4_57379663batch_normalization_4_57379665batch_normalization_4_57379667*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *\
fWRU
S__inference_batch_normalization_4_layer_call_and_return_conditional_losses_573787842/
-batch_normalization_4/StatefulPartitionedCall
re_lu_4/PartitionedCallPartitionedCall6batch_normalization_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_re_lu_4_layer_call_and_return_conditional_losses_573788432
re_lu_4/PartitionedCallή
"upsample_6/StatefulPartitionedCallStatefulPartitionedCall re_lu_4/PartitionedCall:output:0upsample_6_57379671upsample_6_57379673*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,???????????????????????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *Q
fLRJ
H__inference_upsample_6_layer_call_and_return_conditional_losses_573776852$
"upsample_6/StatefulPartitionedCallΠ
$tf_op_layer_concat_6/PartitionedCallPartitionedCall+upsample_6/StatefulPartitionedCall:output:0 re_lu_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *[
fVRT
R__inference_tf_op_layer_concat_6_layer_call_and_return_conditional_losses_573788632&
$tf_op_layer_concat_6/PartitionedCallΕ
conv_6/StatefulPartitionedCallStatefulPartitionedCall-tf_op_layer_concat_6/PartitionedCall:output:0conv_6_57379677conv_6_57379679*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *M
fHRF
D__inference_conv_6_layer_call_and_return_conditional_losses_573788822 
conv_6/StatefulPartitionedCallΜ
-batch_normalization_5/StatefulPartitionedCallStatefulPartitionedCall'conv_6/StatefulPartitionedCall:output:0batch_normalization_5_57379682batch_normalization_5_57379684batch_normalization_5_57379686batch_normalization_5_57379688*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *\
fWRU
S__inference_batch_normalization_5_layer_call_and_return_conditional_losses_573789172/
-batch_normalization_5/StatefulPartitionedCall
re_lu_5/PartitionedCallPartitionedCall6batch_normalization_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_re_lu_5_layer_call_and_return_conditional_losses_573789762
re_lu_5/PartitionedCallέ
"upsample_7/StatefulPartitionedCallStatefulPartitionedCall re_lu_5/PartitionedCall:output:0upsample_7_57379692upsample_7_57379694*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *Q
fLRJ
H__inference_upsample_7_layer_call_and_return_conditional_losses_573778372$
"upsample_7/StatefulPartitionedCallΠ
$tf_op_layer_concat_7/PartitionedCallPartitionedCall+upsample_7/StatefulPartitionedCall:output:0 re_lu_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????0* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *[
fVRT
R__inference_tf_op_layer_concat_7_layer_call_and_return_conditional_losses_573789962&
$tf_op_layer_concat_7/PartitionedCallΔ
conv_7/StatefulPartitionedCallStatefulPartitionedCall-tf_op_layer_concat_7/PartitionedCall:output:0conv_7_57379698conv_7_57379700*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????0@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *M
fHRF
D__inference_conv_7_layer_call_and_return_conditional_losses_573790152 
conv_7/StatefulPartitionedCallΛ
-batch_normalization_6/StatefulPartitionedCallStatefulPartitionedCall'conv_7/StatefulPartitionedCall:output:0batch_normalization_6_57379703batch_normalization_6_57379705batch_normalization_6_57379707batch_normalization_6_57379709*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????0@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *\
fWRU
S__inference_batch_normalization_6_layer_call_and_return_conditional_losses_573790502/
-batch_normalization_6/StatefulPartitionedCall
re_lu_6/PartitionedCallPartitionedCall6batch_normalization_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????0@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_re_lu_6_layer_call_and_return_conditional_losses_573791092
re_lu_6/PartitionedCallέ
"upsample_8/StatefulPartitionedCallStatefulPartitionedCall re_lu_6/PartitionedCall:output:0upsample_8_57379713upsample_8_57379715*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *Q
fLRJ
H__inference_upsample_8_layer_call_and_return_conditional_losses_573779892$
"upsample_8/StatefulPartitionedCallΟ
$tf_op_layer_concat_8/PartitionedCallPartitionedCall+upsample_8/StatefulPartitionedCall:output:0 re_lu_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????0`@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *[
fVRT
R__inference_tf_op_layer_concat_8_layer_call_and_return_conditional_losses_573791292&
$tf_op_layer_concat_8/PartitionedCallΔ
conv_8/StatefulPartitionedCallStatefulPartitionedCall-tf_op_layer_concat_8/PartitionedCall:output:0conv_8_57379719conv_8_57379721*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????0` *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *M
fHRF
D__inference_conv_8_layer_call_and_return_conditional_losses_573791482 
conv_8/StatefulPartitionedCallΛ
-batch_normalization_7/StatefulPartitionedCallStatefulPartitionedCall'conv_8/StatefulPartitionedCall:output:0batch_normalization_7_57379724batch_normalization_7_57379726batch_normalization_7_57379728batch_normalization_7_57379730*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????0` *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *\
fWRU
S__inference_batch_normalization_7_layer_call_and_return_conditional_losses_573791832/
-batch_normalization_7/StatefulPartitionedCall
re_lu_7/PartitionedCallPartitionedCall6batch_normalization_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????0` * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_re_lu_7_layer_call_and_return_conditional_losses_573792422
re_lu_7/PartitionedCallέ
"upsample_9/StatefulPartitionedCallStatefulPartitionedCall re_lu_7/PartitionedCall:output:0upsample_9_57379734upsample_9_57379736*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *Q
fLRJ
H__inference_upsample_9_layer_call_and_return_conditional_losses_573781412$
"upsample_9/StatefulPartitionedCallΞ
$tf_op_layer_concat_9/PartitionedCallPartitionedCall+upsample_9/StatefulPartitionedCall:output:0re_lu/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????`ΐ * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *[
fVRT
R__inference_tf_op_layer_concat_9_layer_call_and_return_conditional_losses_573792622&
$tf_op_layer_concat_9/PartitionedCallΕ
conv_9/StatefulPartitionedCallStatefulPartitionedCall-tf_op_layer_concat_9/PartitionedCall:output:0conv_9_57379740conv_9_57379742*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????`ΐ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *M
fHRF
D__inference_conv_9_layer_call_and_return_conditional_losses_573792812 
conv_9/StatefulPartitionedCallΜ
-batch_normalization_8/StatefulPartitionedCallStatefulPartitionedCall'conv_9/StatefulPartitionedCall:output:0batch_normalization_8_57379745batch_normalization_8_57379747batch_normalization_8_57379749batch_normalization_8_57379751*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????`ΐ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *\
fWRU
S__inference_batch_normalization_8_layer_call_and_return_conditional_losses_573793162/
-batch_normalization_8/StatefulPartitionedCall
re_lu_8/PartitionedCallPartitionedCall6batch_normalization_8/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????`ΐ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_re_lu_8_layer_call_and_return_conditional_losses_573793752
re_lu_8/PartitionedCall³
final/StatefulPartitionedCallStatefulPartitionedCall re_lu_8/PartitionedCall:output:0final_57379755final_57379757*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????`ΐ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *L
fGRE
C__inference_final_layer_call_and_return_conditional_losses_573793932
final/StatefulPartitionedCall
IdentityIdentity&final/StatefulPartitionedCall:output:0,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall.^batch_normalization_3/StatefulPartitionedCall.^batch_normalization_4/StatefulPartitionedCall.^batch_normalization_5/StatefulPartitionedCall.^batch_normalization_6/StatefulPartitionedCall.^batch_normalization_7/StatefulPartitionedCall.^batch_normalization_8/StatefulPartitionedCall^conv_1/StatefulPartitionedCall^conv_2/StatefulPartitionedCall^conv_3/StatefulPartitionedCall^conv_4/StatefulPartitionedCall^conv_5/StatefulPartitionedCall^conv_6/StatefulPartitionedCall^conv_7/StatefulPartitionedCall^conv_8/StatefulPartitionedCall^conv_9/StatefulPartitionedCall^final/StatefulPartitionedCall#^upsample_6/StatefulPartitionedCall#^upsample_7/StatefulPartitionedCall#^upsample_8/StatefulPartitionedCall#^upsample_9/StatefulPartitionedCall*
T0*0
_output_shapes
:?????????`ΐ2

Identity"
identityIdentity:output:0*±
_input_shapes
:?????????`ΐ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2^
-batch_normalization_3/StatefulPartitionedCall-batch_normalization_3/StatefulPartitionedCall2^
-batch_normalization_4/StatefulPartitionedCall-batch_normalization_4/StatefulPartitionedCall2^
-batch_normalization_5/StatefulPartitionedCall-batch_normalization_5/StatefulPartitionedCall2^
-batch_normalization_6/StatefulPartitionedCall-batch_normalization_6/StatefulPartitionedCall2^
-batch_normalization_7/StatefulPartitionedCall-batch_normalization_7/StatefulPartitionedCall2^
-batch_normalization_8/StatefulPartitionedCall-batch_normalization_8/StatefulPartitionedCall2@
conv_1/StatefulPartitionedCallconv_1/StatefulPartitionedCall2@
conv_2/StatefulPartitionedCallconv_2/StatefulPartitionedCall2@
conv_3/StatefulPartitionedCallconv_3/StatefulPartitionedCall2@
conv_4/StatefulPartitionedCallconv_4/StatefulPartitionedCall2@
conv_5/StatefulPartitionedCallconv_5/StatefulPartitionedCall2@
conv_6/StatefulPartitionedCallconv_6/StatefulPartitionedCall2@
conv_7/StatefulPartitionedCallconv_7/StatefulPartitionedCall2@
conv_8/StatefulPartitionedCallconv_8/StatefulPartitionedCall2@
conv_9/StatefulPartitionedCallconv_9/StatefulPartitionedCall2>
final/StatefulPartitionedCallfinal/StatefulPartitionedCall2H
"upsample_6/StatefulPartitionedCall"upsample_6/StatefulPartitionedCall2H
"upsample_7/StatefulPartitionedCall"upsample_7/StatefulPartitionedCall2H
"upsample_8/StatefulPartitionedCall"upsample_8/StatefulPartitionedCall2H
"upsample_9/StatefulPartitionedCall"upsample_9/StatefulPartitionedCall:X T
0
_output_shapes
:?????????`ΐ
 
_user_specified_nameinputs

°
S__inference_batch_normalization_2_layer_call_and_return_conditional_losses_57378558

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity’AssignNewValue’AssignNewValue_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ψ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????0@:@:@:@:@:*
epsilon%o:*
exponential_avg_factor%
Χ#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*/
_output_shapes
:?????????0@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????0@::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:W S
/
_output_shapes
:?????????0@
 
_user_specified_nameinputs
Υ
R
6__inference_tf_op_layer_RealDiv_layer_call_fn_57381211

inputs
identityέ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????`ΐ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *Z
fURS
Q__inference_tf_op_layer_RealDiv_layer_call_and_return_conditional_losses_573782652
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:?????????`ΐ2

Identity"
identityIdentity:output:0*/
_input_shapes
:?????????`ΐ:X T
0
_output_shapes
:?????????`ΐ
 
_user_specified_nameinputs

μ#
$__inference__traced_restore_57383123
file_prefix"
assignvariableop_conv_1_kernel"
assignvariableop_1_conv_1_bias0
,assignvariableop_2_batch_normalization_gamma/
+assignvariableop_3_batch_normalization_beta6
2assignvariableop_4_batch_normalization_moving_mean:
6assignvariableop_5_batch_normalization_moving_variance$
 assignvariableop_6_conv_2_kernel"
assignvariableop_7_conv_2_bias2
.assignvariableop_8_batch_normalization_1_gamma1
-assignvariableop_9_batch_normalization_1_beta9
5assignvariableop_10_batch_normalization_1_moving_mean=
9assignvariableop_11_batch_normalization_1_moving_variance%
!assignvariableop_12_conv_3_kernel#
assignvariableop_13_conv_3_bias3
/assignvariableop_14_batch_normalization_2_gamma2
.assignvariableop_15_batch_normalization_2_beta9
5assignvariableop_16_batch_normalization_2_moving_mean=
9assignvariableop_17_batch_normalization_2_moving_variance%
!assignvariableop_18_conv_4_kernel#
assignvariableop_19_conv_4_bias3
/assignvariableop_20_batch_normalization_3_gamma2
.assignvariableop_21_batch_normalization_3_beta9
5assignvariableop_22_batch_normalization_3_moving_mean=
9assignvariableop_23_batch_normalization_3_moving_variance%
!assignvariableop_24_conv_5_kernel#
assignvariableop_25_conv_5_bias3
/assignvariableop_26_batch_normalization_4_gamma2
.assignvariableop_27_batch_normalization_4_beta9
5assignvariableop_28_batch_normalization_4_moving_mean=
9assignvariableop_29_batch_normalization_4_moving_variance)
%assignvariableop_30_upsample_6_kernel'
#assignvariableop_31_upsample_6_bias%
!assignvariableop_32_conv_6_kernel#
assignvariableop_33_conv_6_bias3
/assignvariableop_34_batch_normalization_5_gamma2
.assignvariableop_35_batch_normalization_5_beta9
5assignvariableop_36_batch_normalization_5_moving_mean=
9assignvariableop_37_batch_normalization_5_moving_variance)
%assignvariableop_38_upsample_7_kernel'
#assignvariableop_39_upsample_7_bias%
!assignvariableop_40_conv_7_kernel#
assignvariableop_41_conv_7_bias3
/assignvariableop_42_batch_normalization_6_gamma2
.assignvariableop_43_batch_normalization_6_beta9
5assignvariableop_44_batch_normalization_6_moving_mean=
9assignvariableop_45_batch_normalization_6_moving_variance)
%assignvariableop_46_upsample_8_kernel'
#assignvariableop_47_upsample_8_bias%
!assignvariableop_48_conv_8_kernel#
assignvariableop_49_conv_8_bias3
/assignvariableop_50_batch_normalization_7_gamma2
.assignvariableop_51_batch_normalization_7_beta9
5assignvariableop_52_batch_normalization_7_moving_mean=
9assignvariableop_53_batch_normalization_7_moving_variance)
%assignvariableop_54_upsample_9_kernel'
#assignvariableop_55_upsample_9_bias%
!assignvariableop_56_conv_9_kernel#
assignvariableop_57_conv_9_bias3
/assignvariableop_58_batch_normalization_8_gamma2
.assignvariableop_59_batch_normalization_8_beta9
5assignvariableop_60_batch_normalization_8_moving_mean=
9assignvariableop_61_batch_normalization_8_moving_variance$
 assignvariableop_62_final_kernel"
assignvariableop_63_final_bias
identity_65’AssignVariableOp’AssignVariableOp_1’AssignVariableOp_10’AssignVariableOp_11’AssignVariableOp_12’AssignVariableOp_13’AssignVariableOp_14’AssignVariableOp_15’AssignVariableOp_16’AssignVariableOp_17’AssignVariableOp_18’AssignVariableOp_19’AssignVariableOp_2’AssignVariableOp_20’AssignVariableOp_21’AssignVariableOp_22’AssignVariableOp_23’AssignVariableOp_24’AssignVariableOp_25’AssignVariableOp_26’AssignVariableOp_27’AssignVariableOp_28’AssignVariableOp_29’AssignVariableOp_3’AssignVariableOp_30’AssignVariableOp_31’AssignVariableOp_32’AssignVariableOp_33’AssignVariableOp_34’AssignVariableOp_35’AssignVariableOp_36’AssignVariableOp_37’AssignVariableOp_38’AssignVariableOp_39’AssignVariableOp_4’AssignVariableOp_40’AssignVariableOp_41’AssignVariableOp_42’AssignVariableOp_43’AssignVariableOp_44’AssignVariableOp_45’AssignVariableOp_46’AssignVariableOp_47’AssignVariableOp_48’AssignVariableOp_49’AssignVariableOp_5’AssignVariableOp_50’AssignVariableOp_51’AssignVariableOp_52’AssignVariableOp_53’AssignVariableOp_54’AssignVariableOp_55’AssignVariableOp_56’AssignVariableOp_57’AssignVariableOp_58’AssignVariableOp_59’AssignVariableOp_6’AssignVariableOp_60’AssignVariableOp_61’AssignVariableOp_62’AssignVariableOp_63’AssignVariableOp_7’AssignVariableOp_8’AssignVariableOp_9
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:A*
dtype0*
valueBAB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-7/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-7/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-9/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-9/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-9/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-12/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-12/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-12/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-12/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-13/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-13/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-14/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-14/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-15/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-15/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-15/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-15/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-16/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-16/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-17/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-17/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-18/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-18/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-18/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-18/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-19/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-19/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-20/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-20/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-21/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-21/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-21/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-21/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-22/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-22/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:A*
dtype0*
valueBAB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slicesσ
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*
_output_shapes
:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*O
dtypesE
C2A2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity
AssignVariableOpAssignVariableOpassignvariableop_conv_1_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1£
AssignVariableOp_1AssignVariableOpassignvariableop_1_conv_1_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2±
AssignVariableOp_2AssignVariableOp,assignvariableop_2_batch_normalization_gammaIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3°
AssignVariableOp_3AssignVariableOp+assignvariableop_3_batch_normalization_betaIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4·
AssignVariableOp_4AssignVariableOp2assignvariableop_4_batch_normalization_moving_meanIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5»
AssignVariableOp_5AssignVariableOp6assignvariableop_5_batch_normalization_moving_varianceIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6₯
AssignVariableOp_6AssignVariableOp assignvariableop_6_conv_2_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7£
AssignVariableOp_7AssignVariableOpassignvariableop_7_conv_2_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8³
AssignVariableOp_8AssignVariableOp.assignvariableop_8_batch_normalization_1_gammaIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9²
AssignVariableOp_9AssignVariableOp-assignvariableop_9_batch_normalization_1_betaIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10½
AssignVariableOp_10AssignVariableOp5assignvariableop_10_batch_normalization_1_moving_meanIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11Α
AssignVariableOp_11AssignVariableOp9assignvariableop_11_batch_normalization_1_moving_varianceIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12©
AssignVariableOp_12AssignVariableOp!assignvariableop_12_conv_3_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13§
AssignVariableOp_13AssignVariableOpassignvariableop_13_conv_3_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14·
AssignVariableOp_14AssignVariableOp/assignvariableop_14_batch_normalization_2_gammaIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15Ά
AssignVariableOp_15AssignVariableOp.assignvariableop_15_batch_normalization_2_betaIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16½
AssignVariableOp_16AssignVariableOp5assignvariableop_16_batch_normalization_2_moving_meanIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17Α
AssignVariableOp_17AssignVariableOp9assignvariableop_17_batch_normalization_2_moving_varianceIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18©
AssignVariableOp_18AssignVariableOp!assignvariableop_18_conv_4_kernelIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19§
AssignVariableOp_19AssignVariableOpassignvariableop_19_conv_4_biasIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20·
AssignVariableOp_20AssignVariableOp/assignvariableop_20_batch_normalization_3_gammaIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21Ά
AssignVariableOp_21AssignVariableOp.assignvariableop_21_batch_normalization_3_betaIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22½
AssignVariableOp_22AssignVariableOp5assignvariableop_22_batch_normalization_3_moving_meanIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23Α
AssignVariableOp_23AssignVariableOp9assignvariableop_23_batch_normalization_3_moving_varianceIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24©
AssignVariableOp_24AssignVariableOp!assignvariableop_24_conv_5_kernelIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25§
AssignVariableOp_25AssignVariableOpassignvariableop_25_conv_5_biasIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26·
AssignVariableOp_26AssignVariableOp/assignvariableop_26_batch_normalization_4_gammaIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27Ά
AssignVariableOp_27AssignVariableOp.assignvariableop_27_batch_normalization_4_betaIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28½
AssignVariableOp_28AssignVariableOp5assignvariableop_28_batch_normalization_4_moving_meanIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29Α
AssignVariableOp_29AssignVariableOp9assignvariableop_29_batch_normalization_4_moving_varianceIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30­
AssignVariableOp_30AssignVariableOp%assignvariableop_30_upsample_6_kernelIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31«
AssignVariableOp_31AssignVariableOp#assignvariableop_31_upsample_6_biasIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32©
AssignVariableOp_32AssignVariableOp!assignvariableop_32_conv_6_kernelIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33§
AssignVariableOp_33AssignVariableOpassignvariableop_33_conv_6_biasIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34·
AssignVariableOp_34AssignVariableOp/assignvariableop_34_batch_normalization_5_gammaIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35Ά
AssignVariableOp_35AssignVariableOp.assignvariableop_35_batch_normalization_5_betaIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36½
AssignVariableOp_36AssignVariableOp5assignvariableop_36_batch_normalization_5_moving_meanIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37Α
AssignVariableOp_37AssignVariableOp9assignvariableop_37_batch_normalization_5_moving_varianceIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38­
AssignVariableOp_38AssignVariableOp%assignvariableop_38_upsample_7_kernelIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39«
AssignVariableOp_39AssignVariableOp#assignvariableop_39_upsample_7_biasIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40©
AssignVariableOp_40AssignVariableOp!assignvariableop_40_conv_7_kernelIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41§
AssignVariableOp_41AssignVariableOpassignvariableop_41_conv_7_biasIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42·
AssignVariableOp_42AssignVariableOp/assignvariableop_42_batch_normalization_6_gammaIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43Ά
AssignVariableOp_43AssignVariableOp.assignvariableop_43_batch_normalization_6_betaIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_43n
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:2
Identity_44½
AssignVariableOp_44AssignVariableOp5assignvariableop_44_batch_normalization_6_moving_meanIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_44n
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:2
Identity_45Α
AssignVariableOp_45AssignVariableOp9assignvariableop_45_batch_normalization_6_moving_varianceIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_45n
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:2
Identity_46­
AssignVariableOp_46AssignVariableOp%assignvariableop_46_upsample_8_kernelIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_46n
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:2
Identity_47«
AssignVariableOp_47AssignVariableOp#assignvariableop_47_upsample_8_biasIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_47n
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:2
Identity_48©
AssignVariableOp_48AssignVariableOp!assignvariableop_48_conv_8_kernelIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_48n
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:2
Identity_49§
AssignVariableOp_49AssignVariableOpassignvariableop_49_conv_8_biasIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_49n
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:2
Identity_50·
AssignVariableOp_50AssignVariableOp/assignvariableop_50_batch_normalization_7_gammaIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_50n
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:2
Identity_51Ά
AssignVariableOp_51AssignVariableOp.assignvariableop_51_batch_normalization_7_betaIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_51n
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:2
Identity_52½
AssignVariableOp_52AssignVariableOp5assignvariableop_52_batch_normalization_7_moving_meanIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_52n
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:2
Identity_53Α
AssignVariableOp_53AssignVariableOp9assignvariableop_53_batch_normalization_7_moving_varianceIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_53n
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:2
Identity_54­
AssignVariableOp_54AssignVariableOp%assignvariableop_54_upsample_9_kernelIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_54n
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:2
Identity_55«
AssignVariableOp_55AssignVariableOp#assignvariableop_55_upsample_9_biasIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_55n
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:2
Identity_56©
AssignVariableOp_56AssignVariableOp!assignvariableop_56_conv_9_kernelIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_56n
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:2
Identity_57§
AssignVariableOp_57AssignVariableOpassignvariableop_57_conv_9_biasIdentity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_57n
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:2
Identity_58·
AssignVariableOp_58AssignVariableOp/assignvariableop_58_batch_normalization_8_gammaIdentity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_58n
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:2
Identity_59Ά
AssignVariableOp_59AssignVariableOp.assignvariableop_59_batch_normalization_8_betaIdentity_59:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_59n
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:2
Identity_60½
AssignVariableOp_60AssignVariableOp5assignvariableop_60_batch_normalization_8_moving_meanIdentity_60:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_60n
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:2
Identity_61Α
AssignVariableOp_61AssignVariableOp9assignvariableop_61_batch_normalization_8_moving_varianceIdentity_61:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_61n
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:2
Identity_62¨
AssignVariableOp_62AssignVariableOp assignvariableop_62_final_kernelIdentity_62:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_62n
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:2
Identity_63¦
AssignVariableOp_63AssignVariableOpassignvariableop_63_final_biasIdentity_63:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_639
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOpή
Identity_64Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_64Ρ
Identity_65IdentityIdentity_64:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_65"#
identity_65Identity_65:output:0*
_input_shapes
: ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::2$
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
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_54AssignVariableOp_542*
AssignVariableOp_55AssignVariableOp_552*
AssignVariableOp_56AssignVariableOp_562*
AssignVariableOp_57AssignVariableOp_572*
AssignVariableOp_58AssignVariableOp_582*
AssignVariableOp_59AssignVariableOp_592(
AssignVariableOp_6AssignVariableOp_62*
AssignVariableOp_60AssignVariableOp_602*
AssignVariableOp_61AssignVariableOp_612*
AssignVariableOp_62AssignVariableOp_622*
AssignVariableOp_63AssignVariableOp_632(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix

°
S__inference_batch_normalization_7_layer_call_and_return_conditional_losses_57382463

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity’AssignNewValue’AssignNewValue_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ψ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????0` : : : : :*
epsilon%o:*
exponential_avg_factor%
Χ#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*/
_output_shapes
:?????????0` 2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????0` ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:W S
/
_output_shapes
:?????????0` 
 
_user_specified_nameinputs

~
)__inference_conv_2_layer_call_fn_57381398

inputs
unknown
	unknown_0
identity’StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????0` *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *M
fHRF
D__inference_conv_2_layer_call_and_return_conditional_losses_573784102
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????0` 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????0`::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????0`
 
_user_specified_nameinputs

g
K__inference_max_pooling2d_layer_call_and_return_conditional_losses_57377189

inputs
identity­
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
2	
MaxPool
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs

c
7__inference_tf_op_layer_concat_8_layer_call_fn_57382360
inputs_0
inputs_1
identityκ
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????0`@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *[
fVRT
R__inference_tf_op_layer_concat_8_layer_call_and_return_conditional_losses_573791292
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????0`@2

Identity"
identityIdentity:output:0*[
_input_shapesJ
H:+??????????????????????????? :?????????0` :k g
A
_output_shapes/
-:+??????????????????????????? 
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:?????????0` 
"
_user_specified_name
inputs/1
¨
«
8__inference_batch_normalization_7_layer_call_fn_57382443

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity’StatefulPartitionedCallΌ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? *&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *\
fWRU
S__inference_batch_normalization_7_layer_call_and_return_conditional_losses_573780922
StatefulPartitionedCall¨
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+??????????????????????????? ::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs

~
)__inference_conv_8_layer_call_fn_57382379

inputs
unknown
	unknown_0
identity’StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????0` *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *M
fHRF
D__inference_conv_8_layer_call_and_return_conditional_losses_573791482
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????0` 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????0`@::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????0`@
 
_user_specified_nameinputs
ΰ
«
8__inference_batch_normalization_6_layer_call_fn_57382273

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity’StatefulPartitionedCallͺ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????0@*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *\
fWRU
S__inference_batch_normalization_6_layer_call_and_return_conditional_losses_573790682
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????0@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????0@::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????0@
 
_user_specified_nameinputs


Q__inference_batch_normalization_layer_call_and_return_conditional_losses_57377172

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ά
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o:*
is_training( 2
FusedBatchNormV3
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????:::::i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
½
F
*__inference_re_lu_4_layer_call_fn_57382007

inputs
identityΡ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_re_lu_4_layer_call_and_return_conditional_losses_573788432
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:?????????:X T
0
_output_shapes
:?????????
 
_user_specified_nameinputs
Θ$
Ί
H__inference_upsample_8_layer_call_and_return_conditional_losses_57377989

inputs,
(conv2d_transpose_readvariableop_resource#
biasadd_readvariableop_resource
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2β
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2μ
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1x
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2μ
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_2P
mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
mul/y\
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: 2
mulP
add/yConst*
_output_shapes
: *
dtype0*
value	B : 2
add/yM
addAddV2mul:z:0add/y:output:0*
T0*
_output_shapes
: 2
addT
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2	
mul_1/yb
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: 2
mul_1T
add_1/yConst*
_output_shapes
: *
dtype0*
value	B : 2	
add_1/yU
add_1AddV2	mul_1:z:0add_1/y:output:0*
T0*
_output_shapes
: 2
add_1T
stack/3Const*
_output_shapes
: *
dtype0*
value	B : 2	
stack/3
stackPackstrided_slice:output:0add:z:0	add_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:2
stackx
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2μ
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_3³
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
: @*
dtype02!
conv2d_transpose/ReadVariableOpρ
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+??????????????????????????? *
paddingVALID*
strides
2
conv2d_transpose
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp€
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+??????????????????????????? 2	
BiasAdd~
IdentityIdentityBiasAdd:output:0*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+???????????????????????????@:::i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
Λ
°
S__inference_batch_normalization_1_layer_call_and_return_conditional_losses_57377257

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity’AssignNewValue’AssignNewValue_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1κ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+??????????????????????????? : : : : :*
epsilon%o:*
exponential_avg_factor%
Χ#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1¦
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+??????????????????????????? ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
Υ
_
C__inference_re_lu_layer_call_and_return_conditional_losses_57381374

inputs
identityW
ReluReluinputs*
T0*0
_output_shapes
:?????????`ΐ2
Reluo
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:?????????`ΐ2

Identity"
identityIdentity:output:0*/
_input_shapes
:?????????`ΐ:X T
0
_output_shapes
:?????????`ΐ
 
_user_specified_nameinputs

~
)__inference_conv_4_layer_call_fn_57381712

inputs
unknown
	unknown_0
identity’StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *M
fHRF
D__inference_conv_4_layer_call_and_return_conditional_losses_573786362
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????@::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
§

S__inference_batch_normalization_4_layer_call_and_return_conditional_losses_57377636

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityu
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype02
ReadVariableOp_1¨
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1α
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,???????????????????????????:::::*
epsilon%o:*
is_training( 2
FusedBatchNormV3
IdentityIdentityFusedBatchNormV3:y:0*
T0*B
_output_shapes0
.:,???????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,???????????????????????????:::::j f
B
_output_shapes0
.:,???????????????????????????
 
_user_specified_nameinputs
¦
«
8__inference_batch_normalization_7_layer_call_fn_57382430

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity’StatefulPartitionedCallΊ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *\
fWRU
S__inference_batch_normalization_7_layer_call_and_return_conditional_losses_573780612
StatefulPartitionedCall¨
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+??????????????????????????? ::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
Ά
N
2__inference_max_pooling2d_1_layer_call_fn_57377311

inputs
identityσ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *V
fQRO
M__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_573773052
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
§

S__inference_batch_normalization_4_layer_call_and_return_conditional_losses_57381971

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityu
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype02
ReadVariableOp_1¨
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1α
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,???????????????????????????:::::*
epsilon%o:*
is_training( 2
FusedBatchNormV3
IdentityIdentityFusedBatchNormV3:y:0*
T0*B
_output_shapes0
.:,???????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,???????????????????????????:::::j f
B
_output_shapes0
.:,???????????????????????????
 
_user_specified_nameinputs
ή
«
8__inference_batch_normalization_2_layer_call_fn_57381606

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity’StatefulPartitionedCall¨
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????0@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *\
fWRU
S__inference_batch_normalization_2_layer_call_and_return_conditional_losses_573785582
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????0@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????0@::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????0@
 
_user_specified_nameinputs
Τ

Q__inference_batch_normalization_layer_call_and_return_conditional_losses_57378350

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Λ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*L
_output_shapes:
8:?????????`ΐ:::::*
epsilon%o:*
is_training( 2
FusedBatchNormV3q
IdentityIdentityFusedBatchNormV3:y:0*
T0*0
_output_shapes
:?????????`ΐ2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:?????????`ΐ:::::X T
0
_output_shapes
:?????????`ΐ
 
_user_specified_nameinputs


S__inference_batch_normalization_2_layer_call_and_return_conditional_losses_57377404

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ά
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%o:*
is_training( 2
FusedBatchNormV3
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????@:::::i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs"ΈL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*ΐ
serving_default¬
J

imageInput<
serving_default_imageInput:0?????????`ΐB
final9
StatefulPartitionedCall:0?????????`ΐtensorflow/serving/predict:ρΡ

½ν
layer-0
layer-1
layer-2
layer_with_weights-0
layer-3
layer_with_weights-1
layer-4
layer-5
layer-6
layer_with_weights-2
layer-7
	layer_with_weights-3
	layer-8

layer-9
layer-10
layer_with_weights-4
layer-11
layer_with_weights-5
layer-12
layer-13
layer-14
layer_with_weights-6
layer-15
layer_with_weights-7
layer-16
layer-17
layer-18
layer_with_weights-8
layer-19
layer_with_weights-9
layer-20
layer-21
layer_with_weights-10
layer-22
layer-23
layer_with_weights-11
layer-24
layer_with_weights-12
layer-25
layer-26
layer_with_weights-13
layer-27
layer-28
layer_with_weights-14
layer-29
layer_with_weights-15
layer-30
 layer-31
!layer_with_weights-16
!layer-32
"layer-33
#layer_with_weights-17
#layer-34
$layer_with_weights-18
$layer-35
%layer-36
&layer_with_weights-19
&layer-37
'layer-38
(layer_with_weights-20
(layer-39
)layer_with_weights-21
)layer-40
*layer-41
+layer_with_weights-22
+layer-42
,regularization_losses
-	variables
.trainable_variables
/	keras_api
0
signatures
ω__call__
+ϊ&call_and_return_all_conditional_losses
ϋ_default_save_signature"Κβ
_tf_keras_network­β{"class_name": "Functional", "name": "functional_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "functional_1", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 96, 192, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "imageInput"}, "name": "imageInput", "inbound_nodes": []}, {"class_name": "TensorFlowOpLayer", "config": {"name": "RealDiv", "trainable": true, "dtype": "float32", "node_def": {"name": "RealDiv", "op": "RealDiv", "input": ["imageInput", "RealDiv/y"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {"1": 127.5}}, "name": "tf_op_layer_RealDiv", "inbound_nodes": [[["imageInput", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "Sub", "trainable": true, "dtype": "float32", "node_def": {"name": "Sub", "op": "Sub", "input": ["RealDiv", "Sub/y"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {"1": 1.0}}, "name": "tf_op_layer_Sub", "inbound_nodes": [[["tf_op_layer_RealDiv", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv_1", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv_1", "inbound_nodes": [[["tf_op_layer_Sub", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization", "inbound_nodes": [[["conv_1", 0, 0, {}]]]}, {"class_name": "ReLU", "config": {"name": "re_lu", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}, "name": "re_lu", "inbound_nodes": [[["batch_normalization", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "max_pooling2d", "inbound_nodes": [[["re_lu", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv_2", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv_2", "inbound_nodes": [[["max_pooling2d", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_1", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_1", "inbound_nodes": [[["conv_2", 0, 0, {}]]]}, {"class_name": "ReLU", "config": {"name": "re_lu_1", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}, "name": "re_lu_1", "inbound_nodes": [[["batch_normalization_1", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_1", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "max_pooling2d_1", "inbound_nodes": [[["re_lu_1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv_3", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv_3", "inbound_nodes": [[["max_pooling2d_1", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_2", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_2", "inbound_nodes": [[["conv_3", 0, 0, {}]]]}, {"class_name": "ReLU", "config": {"name": "re_lu_2", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}, "name": "re_lu_2", "inbound_nodes": [[["batch_normalization_2", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_2", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "max_pooling2d_2", "inbound_nodes": [[["re_lu_2", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv_4", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv_4", "inbound_nodes": [[["max_pooling2d_2", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_3", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_3", "inbound_nodes": [[["conv_4", 0, 0, {}]]]}, {"class_name": "ReLU", "config": {"name": "re_lu_3", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}, "name": "re_lu_3", "inbound_nodes": [[["batch_normalization_3", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_3", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "max_pooling2d_3", "inbound_nodes": [[["re_lu_3", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv_5", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv_5", "inbound_nodes": [[["max_pooling2d_3", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_4", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_4", "inbound_nodes": [[["conv_5", 0, 0, {}]]]}, {"class_name": "ReLU", "config": {"name": "re_lu_4", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}, "name": "re_lu_4", "inbound_nodes": [[["batch_normalization_4", 0, 0, {}]]]}, {"class_name": "Conv2DTranspose", "config": {"name": "upsample_6", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [2, 2]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "name": "upsample_6", "inbound_nodes": [[["re_lu_4", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "concat_6", "trainable": true, "dtype": "float32", "node_def": {"name": "concat_6", "op": "ConcatV2", "input": ["upsample_6/BiasAdd", "re_lu_3/Relu", "concat_6/axis"], "attr": {"Tidx": {"type": "DT_INT32"}, "T": {"type": "DT_FLOAT"}, "N": {"i": "2"}}}, "constants": {"2": -1}}, "name": "tf_op_layer_concat_6", "inbound_nodes": [[["upsample_6", 0, 0, {}], ["re_lu_3", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv_6", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv_6", "inbound_nodes": [[["tf_op_layer_concat_6", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_5", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_5", "inbound_nodes": [[["conv_6", 0, 0, {}]]]}, {"class_name": "ReLU", "config": {"name": "re_lu_5", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}, "name": "re_lu_5", "inbound_nodes": [[["batch_normalization_5", 0, 0, {}]]]}, {"class_name": "Conv2DTranspose", "config": {"name": "upsample_7", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [2, 2]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "name": "upsample_7", "inbound_nodes": [[["re_lu_5", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "concat_7", "trainable": true, "dtype": "float32", "node_def": {"name": "concat_7", "op": "ConcatV2", "input": ["upsample_7/BiasAdd", "re_lu_2/Relu", "concat_7/axis"], "attr": {"Tidx": {"type": "DT_INT32"}, "T": {"type": "DT_FLOAT"}, "N": {"i": "2"}}}, "constants": {"2": -1}}, "name": "tf_op_layer_concat_7", "inbound_nodes": [[["upsample_7", 0, 0, {}], ["re_lu_2", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv_7", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv_7", "inbound_nodes": [[["tf_op_layer_concat_7", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_6", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_6", "inbound_nodes": [[["conv_7", 0, 0, {}]]]}, {"class_name": "ReLU", "config": {"name": "re_lu_6", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}, "name": "re_lu_6", "inbound_nodes": [[["batch_normalization_6", 0, 0, {}]]]}, {"class_name": "Conv2DTranspose", "config": {"name": "upsample_8", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [2, 2]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "name": "upsample_8", "inbound_nodes": [[["re_lu_6", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "concat_8", "trainable": true, "dtype": "float32", "node_def": {"name": "concat_8", "op": "ConcatV2", "input": ["upsample_8/BiasAdd", "re_lu_1/Relu", "concat_8/axis"], "attr": {"Tidx": {"type": "DT_INT32"}, "T": {"type": "DT_FLOAT"}, "N": {"i": "2"}}}, "constants": {"2": -1}}, "name": "tf_op_layer_concat_8", "inbound_nodes": [[["upsample_8", 0, 0, {}], ["re_lu_1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv_8", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv_8", "inbound_nodes": [[["tf_op_layer_concat_8", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_7", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_7", "inbound_nodes": [[["conv_8", 0, 0, {}]]]}, {"class_name": "ReLU", "config": {"name": "re_lu_7", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}, "name": "re_lu_7", "inbound_nodes": [[["batch_normalization_7", 0, 0, {}]]]}, {"class_name": "Conv2DTranspose", "config": {"name": "upsample_9", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [2, 2]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "name": "upsample_9", "inbound_nodes": [[["re_lu_7", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "concat_9", "trainable": true, "dtype": "float32", "node_def": {"name": "concat_9", "op": "ConcatV2", "input": ["upsample_9/BiasAdd", "re_lu/Relu", "concat_9/axis"], "attr": {"Tidx": {"type": "DT_INT32"}, "T": {"type": "DT_FLOAT"}, "N": {"i": "2"}}}, "constants": {"2": -1}}, "name": "tf_op_layer_concat_9", "inbound_nodes": [[["upsample_9", 0, 0, {}], ["re_lu", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv_9", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv_9", "inbound_nodes": [[["tf_op_layer_concat_9", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_8", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_8", "inbound_nodes": [[["conv_9", 0, 0, {}]]]}, {"class_name": "ReLU", "config": {"name": "re_lu_8", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}, "name": "re_lu_8", "inbound_nodes": [[["batch_normalization_8", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "final", "trainable": true, "dtype": "float32", "filters": 2, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "final", "inbound_nodes": [[["re_lu_8", 0, 0, {}]]]}], "input_layers": [["imageInput", 0, 0]], "output_layers": [["final", 0, 0]]}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 96, 192, 3]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "functional_1", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 96, 192, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "imageInput"}, "name": "imageInput", "inbound_nodes": []}, {"class_name": "TensorFlowOpLayer", "config": {"name": "RealDiv", "trainable": true, "dtype": "float32", "node_def": {"name": "RealDiv", "op": "RealDiv", "input": ["imageInput", "RealDiv/y"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {"1": 127.5}}, "name": "tf_op_layer_RealDiv", "inbound_nodes": [[["imageInput", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "Sub", "trainable": true, "dtype": "float32", "node_def": {"name": "Sub", "op": "Sub", "input": ["RealDiv", "Sub/y"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {"1": 1.0}}, "name": "tf_op_layer_Sub", "inbound_nodes": [[["tf_op_layer_RealDiv", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv_1", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv_1", "inbound_nodes": [[["tf_op_layer_Sub", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization", "inbound_nodes": [[["conv_1", 0, 0, {}]]]}, {"class_name": "ReLU", "config": {"name": "re_lu", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}, "name": "re_lu", "inbound_nodes": [[["batch_normalization", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "max_pooling2d", "inbound_nodes": [[["re_lu", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv_2", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv_2", "inbound_nodes": [[["max_pooling2d", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_1", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_1", "inbound_nodes": [[["conv_2", 0, 0, {}]]]}, {"class_name": "ReLU", "config": {"name": "re_lu_1", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}, "name": "re_lu_1", "inbound_nodes": [[["batch_normalization_1", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_1", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "max_pooling2d_1", "inbound_nodes": [[["re_lu_1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv_3", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv_3", "inbound_nodes": [[["max_pooling2d_1", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_2", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_2", "inbound_nodes": [[["conv_3", 0, 0, {}]]]}, {"class_name": "ReLU", "config": {"name": "re_lu_2", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}, "name": "re_lu_2", "inbound_nodes": [[["batch_normalization_2", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_2", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "max_pooling2d_2", "inbound_nodes": [[["re_lu_2", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv_4", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv_4", "inbound_nodes": [[["max_pooling2d_2", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_3", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_3", "inbound_nodes": [[["conv_4", 0, 0, {}]]]}, {"class_name": "ReLU", "config": {"name": "re_lu_3", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}, "name": "re_lu_3", "inbound_nodes": [[["batch_normalization_3", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_3", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "max_pooling2d_3", "inbound_nodes": [[["re_lu_3", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv_5", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv_5", "inbound_nodes": [[["max_pooling2d_3", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_4", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_4", "inbound_nodes": [[["conv_5", 0, 0, {}]]]}, {"class_name": "ReLU", "config": {"name": "re_lu_4", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}, "name": "re_lu_4", "inbound_nodes": [[["batch_normalization_4", 0, 0, {}]]]}, {"class_name": "Conv2DTranspose", "config": {"name": "upsample_6", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [2, 2]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "name": "upsample_6", "inbound_nodes": [[["re_lu_4", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "concat_6", "trainable": true, "dtype": "float32", "node_def": {"name": "concat_6", "op": "ConcatV2", "input": ["upsample_6/BiasAdd", "re_lu_3/Relu", "concat_6/axis"], "attr": {"Tidx": {"type": "DT_INT32"}, "T": {"type": "DT_FLOAT"}, "N": {"i": "2"}}}, "constants": {"2": -1}}, "name": "tf_op_layer_concat_6", "inbound_nodes": [[["upsample_6", 0, 0, {}], ["re_lu_3", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv_6", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv_6", "inbound_nodes": [[["tf_op_layer_concat_6", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_5", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_5", "inbound_nodes": [[["conv_6", 0, 0, {}]]]}, {"class_name": "ReLU", "config": {"name": "re_lu_5", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}, "name": "re_lu_5", "inbound_nodes": [[["batch_normalization_5", 0, 0, {}]]]}, {"class_name": "Conv2DTranspose", "config": {"name": "upsample_7", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [2, 2]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "name": "upsample_7", "inbound_nodes": [[["re_lu_5", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "concat_7", "trainable": true, "dtype": "float32", "node_def": {"name": "concat_7", "op": "ConcatV2", "input": ["upsample_7/BiasAdd", "re_lu_2/Relu", "concat_7/axis"], "attr": {"Tidx": {"type": "DT_INT32"}, "T": {"type": "DT_FLOAT"}, "N": {"i": "2"}}}, "constants": {"2": -1}}, "name": "tf_op_layer_concat_7", "inbound_nodes": [[["upsample_7", 0, 0, {}], ["re_lu_2", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv_7", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv_7", "inbound_nodes": [[["tf_op_layer_concat_7", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_6", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_6", "inbound_nodes": [[["conv_7", 0, 0, {}]]]}, {"class_name": "ReLU", "config": {"name": "re_lu_6", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}, "name": "re_lu_6", "inbound_nodes": [[["batch_normalization_6", 0, 0, {}]]]}, {"class_name": "Conv2DTranspose", "config": {"name": "upsample_8", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [2, 2]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "name": "upsample_8", "inbound_nodes": [[["re_lu_6", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "concat_8", "trainable": true, "dtype": "float32", "node_def": {"name": "concat_8", "op": "ConcatV2", "input": ["upsample_8/BiasAdd", "re_lu_1/Relu", "concat_8/axis"], "attr": {"Tidx": {"type": "DT_INT32"}, "T": {"type": "DT_FLOAT"}, "N": {"i": "2"}}}, "constants": {"2": -1}}, "name": "tf_op_layer_concat_8", "inbound_nodes": [[["upsample_8", 0, 0, {}], ["re_lu_1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv_8", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv_8", "inbound_nodes": [[["tf_op_layer_concat_8", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_7", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_7", "inbound_nodes": [[["conv_8", 0, 0, {}]]]}, {"class_name": "ReLU", "config": {"name": "re_lu_7", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}, "name": "re_lu_7", "inbound_nodes": [[["batch_normalization_7", 0, 0, {}]]]}, {"class_name": "Conv2DTranspose", "config": {"name": "upsample_9", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [2, 2]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "name": "upsample_9", "inbound_nodes": [[["re_lu_7", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "concat_9", "trainable": true, "dtype": "float32", "node_def": {"name": "concat_9", "op": "ConcatV2", "input": ["upsample_9/BiasAdd", "re_lu/Relu", "concat_9/axis"], "attr": {"Tidx": {"type": "DT_INT32"}, "T": {"type": "DT_FLOAT"}, "N": {"i": "2"}}}, "constants": {"2": -1}}, "name": "tf_op_layer_concat_9", "inbound_nodes": [[["upsample_9", 0, 0, {}], ["re_lu", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv_9", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv_9", "inbound_nodes": [[["tf_op_layer_concat_9", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_8", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_8", "inbound_nodes": [[["conv_9", 0, 0, {}]]]}, {"class_name": "ReLU", "config": {"name": "re_lu_8", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}, "name": "re_lu_8", "inbound_nodes": [[["batch_normalization_8", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "final", "trainable": true, "dtype": "float32", "filters": 2, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "final", "inbound_nodes": [[["re_lu_8", 0, 0, {}]]]}], "input_layers": [["imageInput", 0, 0]], "output_layers": [["final", 0, 0]]}}}
"ώ
_tf_keras_input_layerή{"class_name": "InputLayer", "name": "imageInput", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 96, 192, 3]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 96, 192, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "imageInput"}}
β
1regularization_losses
2	variables
3trainable_variables
4	keras_api
ό__call__
+ύ&call_and_return_all_conditional_losses"Ρ
_tf_keras_layer·{"class_name": "TensorFlowOpLayer", "name": "tf_op_layer_RealDiv", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "RealDiv", "trainable": true, "dtype": "float32", "node_def": {"name": "RealDiv", "op": "RealDiv", "input": ["imageInput", "RealDiv/y"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {"1": 127.5}}}
Ι
5regularization_losses
6	variables
7trainable_variables
8	keras_api
ώ__call__
+?&call_and_return_all_conditional_losses"Έ
_tf_keras_layer{"class_name": "TensorFlowOpLayer", "name": "tf_op_layer_Sub", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "Sub", "trainable": true, "dtype": "float32", "node_def": {"name": "Sub", "op": "Sub", "input": ["RealDiv", "Sub/y"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {"1": 1.0}}}
ρ	

9kernel
:bias
;regularization_losses
<	variables
=trainable_variables
>	keras_api
__call__
+&call_and_return_all_conditional_losses"Κ
_tf_keras_layer°{"class_name": "Conv2D", "name": "conv_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv_1", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 3}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 96, 192, 3]}}
Ή	
?axis
	@gamma
Abeta
Bmoving_mean
Cmoving_variance
Dregularization_losses
E	variables
Ftrainable_variables
G	keras_api
__call__
+&call_and_return_all_conditional_losses"γ
_tf_keras_layerΙ{"class_name": "BatchNormalization", "name": "batch_normalization", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 16}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 96, 192, 16]}}
ι
Hregularization_losses
I	variables
Jtrainable_variables
K	keras_api
__call__
+&call_and_return_all_conditional_losses"Ψ
_tf_keras_layerΎ{"class_name": "ReLU", "name": "re_lu", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "re_lu", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}}
ύ
Lregularization_losses
M	variables
Ntrainable_variables
O	keras_api
__call__
+&call_and_return_all_conditional_losses"μ
_tf_keras_layer?{"class_name": "MaxPooling2D", "name": "max_pooling2d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "max_pooling2d", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
ς	

Pkernel
Qbias
Rregularization_losses
S	variables
Ttrainable_variables
U	keras_api
__call__
+&call_and_return_all_conditional_losses"Λ
_tf_keras_layer±{"class_name": "Conv2D", "name": "conv_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv_2", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 16}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 48, 96, 16]}}
Ό	
Vaxis
	Wgamma
Xbeta
Ymoving_mean
Zmoving_variance
[regularization_losses
\	variables
]trainable_variables
^	keras_api
__call__
+&call_and_return_all_conditional_losses"ζ
_tf_keras_layerΜ{"class_name": "BatchNormalization", "name": "batch_normalization_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_1", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 48, 96, 32]}}
ν
_regularization_losses
`	variables
atrainable_variables
b	keras_api
__call__
+&call_and_return_all_conditional_losses"ά
_tf_keras_layerΒ{"class_name": "ReLU", "name": "re_lu_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "re_lu_1", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}}

cregularization_losses
d	variables
etrainable_variables
f	keras_api
__call__
+&call_and_return_all_conditional_losses"π
_tf_keras_layerΦ{"class_name": "MaxPooling2D", "name": "max_pooling2d_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "max_pooling2d_1", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
ς	

gkernel
hbias
iregularization_losses
j	variables
ktrainable_variables
l	keras_api
__call__
+&call_and_return_all_conditional_losses"Λ
_tf_keras_layer±{"class_name": "Conv2D", "name": "conv_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv_3", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 24, 48, 32]}}
Ό	
maxis
	ngamma
obeta
pmoving_mean
qmoving_variance
rregularization_losses
s	variables
ttrainable_variables
u	keras_api
__call__
+&call_and_return_all_conditional_losses"ζ
_tf_keras_layerΜ{"class_name": "BatchNormalization", "name": "batch_normalization_2", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_2", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 24, 48, 64]}}
ν
vregularization_losses
w	variables
xtrainable_variables
y	keras_api
__call__
+&call_and_return_all_conditional_losses"ά
_tf_keras_layerΒ{"class_name": "ReLU", "name": "re_lu_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "re_lu_2", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}}

zregularization_losses
{	variables
|trainable_variables
}	keras_api
__call__
+&call_and_return_all_conditional_losses"π
_tf_keras_layerΦ{"class_name": "MaxPooling2D", "name": "max_pooling2d_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "max_pooling2d_2", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
χ	

~kernel
bias
regularization_losses
	variables
trainable_variables
	keras_api
__call__
+&call_and_return_all_conditional_losses"Μ
_tf_keras_layer²{"class_name": "Conv2D", "name": "conv_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv_4", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 12, 24, 64]}}
Η	
	axis

gamma
	beta
moving_mean
moving_variance
regularization_losses
	variables
trainable_variables
	keras_api
__call__
+&call_and_return_all_conditional_losses"θ
_tf_keras_layerΞ{"class_name": "BatchNormalization", "name": "batch_normalization_3", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_3", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 12, 24, 128]}}
ρ
regularization_losses
	variables
trainable_variables
	keras_api
__call__
+&call_and_return_all_conditional_losses"ά
_tf_keras_layerΒ{"class_name": "ReLU", "name": "re_lu_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "re_lu_3", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}}

regularization_losses
	variables
trainable_variables
	keras_api
__call__
+&call_and_return_all_conditional_losses"π
_tf_keras_layerΦ{"class_name": "MaxPooling2D", "name": "max_pooling2d_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "max_pooling2d_3", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
ϊ	
kernel
	bias
regularization_losses
	variables
trainable_variables
	keras_api
 __call__
+‘&call_and_return_all_conditional_losses"Ν
_tf_keras_layer³{"class_name": "Conv2D", "name": "conv_5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv_5", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 6, 12, 128]}}
Ζ	
	axis

gamma
	beta
moving_mean
moving_variance
 regularization_losses
‘	variables
’trainable_variables
£	keras_api
’__call__
+£&call_and_return_all_conditional_losses"η
_tf_keras_layerΝ{"class_name": "BatchNormalization", "name": "batch_normalization_4", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_4", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 256}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 6, 12, 256]}}
ρ
€regularization_losses
₯	variables
¦trainable_variables
§	keras_api
€__call__
+₯&call_and_return_all_conditional_losses"ά
_tf_keras_layerΒ{"class_name": "ReLU", "name": "re_lu_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "re_lu_4", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}}
€

¨kernel
	©bias
ͺregularization_losses
«	variables
¬trainable_variables
­	keras_api
¦__call__
+§&call_and_return_all_conditional_losses"χ
_tf_keras_layerέ{"class_name": "Conv2DTranspose", "name": "upsample_6", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "upsample_6", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [2, 2]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 256}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 6, 12, 256]}}
²
?regularization_losses
―	variables
°trainable_variables
±	keras_api
¨__call__
+©&call_and_return_all_conditional_losses"
_tf_keras_layer{"class_name": "TensorFlowOpLayer", "name": "tf_op_layer_concat_6", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "concat_6", "trainable": true, "dtype": "float32", "node_def": {"name": "concat_6", "op": "ConcatV2", "input": ["upsample_6/BiasAdd", "re_lu_3/Relu", "concat_6/axis"], "attr": {"Tidx": {"type": "DT_INT32"}, "T": {"type": "DT_FLOAT"}, "N": {"i": "2"}}}, "constants": {"2": -1}}}
ϋ	
²kernel
	³bias
΄regularization_losses
΅	variables
Άtrainable_variables
·	keras_api
ͺ__call__
+«&call_and_return_all_conditional_losses"Ξ
_tf_keras_layer΄{"class_name": "Conv2D", "name": "conv_6", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv_6", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 256}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 12, 24, 256]}}
Η	
	Έaxis

Ήgamma
	Ίbeta
»moving_mean
Όmoving_variance
½regularization_losses
Ύ	variables
Ώtrainable_variables
ΐ	keras_api
¬__call__
+­&call_and_return_all_conditional_losses"θ
_tf_keras_layerΞ{"class_name": "BatchNormalization", "name": "batch_normalization_5", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_5", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 12, 24, 128]}}
ρ
Αregularization_losses
Β	variables
Γtrainable_variables
Δ	keras_api
?__call__
+―&call_and_return_all_conditional_losses"ά
_tf_keras_layerΒ{"class_name": "ReLU", "name": "re_lu_5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "re_lu_5", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}}
€

Εkernel
	Ζbias
Ηregularization_losses
Θ	variables
Ιtrainable_variables
Κ	keras_api
°__call__
+±&call_and_return_all_conditional_losses"χ
_tf_keras_layerέ{"class_name": "Conv2DTranspose", "name": "upsample_7", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "upsample_7", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [2, 2]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 12, 24, 128]}}
²
Λregularization_losses
Μ	variables
Νtrainable_variables
Ξ	keras_api
²__call__
+³&call_and_return_all_conditional_losses"
_tf_keras_layer{"class_name": "TensorFlowOpLayer", "name": "tf_op_layer_concat_7", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "concat_7", "trainable": true, "dtype": "float32", "node_def": {"name": "concat_7", "op": "ConcatV2", "input": ["upsample_7/BiasAdd", "re_lu_2/Relu", "concat_7/axis"], "attr": {"Tidx": {"type": "DT_INT32"}, "T": {"type": "DT_FLOAT"}, "N": {"i": "2"}}}, "constants": {"2": -1}}}
ϊ	
Οkernel
	Πbias
Ρregularization_losses
?	variables
Σtrainable_variables
Τ	keras_api
΄__call__
+΅&call_and_return_all_conditional_losses"Ν
_tf_keras_layer³{"class_name": "Conv2D", "name": "conv_7", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv_7", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 24, 48, 128]}}
Ε	
	Υaxis

Φgamma
	Χbeta
Ψmoving_mean
Ωmoving_variance
Ϊregularization_losses
Ϋ	variables
άtrainable_variables
έ	keras_api
Ά__call__
+·&call_and_return_all_conditional_losses"ζ
_tf_keras_layerΜ{"class_name": "BatchNormalization", "name": "batch_normalization_6", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_6", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 24, 48, 64]}}
ρ
ήregularization_losses
ί	variables
ΰtrainable_variables
α	keras_api
Έ__call__
+Ή&call_and_return_all_conditional_losses"ά
_tf_keras_layerΒ{"class_name": "ReLU", "name": "re_lu_6", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "re_lu_6", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}}
’

βkernel
	γbias
δregularization_losses
ε	variables
ζtrainable_variables
η	keras_api
Ί__call__
+»&call_and_return_all_conditional_losses"υ
_tf_keras_layerΫ{"class_name": "Conv2DTranspose", "name": "upsample_8", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "upsample_8", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [2, 2]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 24, 48, 64]}}
²
θregularization_losses
ι	variables
κtrainable_variables
λ	keras_api
Ό__call__
+½&call_and_return_all_conditional_losses"
_tf_keras_layer{"class_name": "TensorFlowOpLayer", "name": "tf_op_layer_concat_8", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "concat_8", "trainable": true, "dtype": "float32", "node_def": {"name": "concat_8", "op": "ConcatV2", "input": ["upsample_8/BiasAdd", "re_lu_1/Relu", "concat_8/axis"], "attr": {"Tidx": {"type": "DT_INT32"}, "T": {"type": "DT_FLOAT"}, "N": {"i": "2"}}}, "constants": {"2": -1}}}
ψ	
μkernel
	νbias
ξregularization_losses
ο	variables
πtrainable_variables
ρ	keras_api
Ύ__call__
+Ώ&call_and_return_all_conditional_losses"Λ
_tf_keras_layer±{"class_name": "Conv2D", "name": "conv_8", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv_8", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 48, 96, 64]}}
Ε	
	ςaxis

σgamma
	τbeta
υmoving_mean
φmoving_variance
χregularization_losses
ψ	variables
ωtrainable_variables
ϊ	keras_api
ΐ__call__
+Α&call_and_return_all_conditional_losses"ζ
_tf_keras_layerΜ{"class_name": "BatchNormalization", "name": "batch_normalization_7", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_7", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 48, 96, 32]}}
ρ
ϋregularization_losses
ό	variables
ύtrainable_variables
ώ	keras_api
Β__call__
+Γ&call_and_return_all_conditional_losses"ά
_tf_keras_layerΒ{"class_name": "ReLU", "name": "re_lu_7", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "re_lu_7", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}}
’

?kernel
	bias
regularization_losses
	variables
trainable_variables
	keras_api
Δ__call__
+Ε&call_and_return_all_conditional_losses"υ
_tf_keras_layerΫ{"class_name": "Conv2DTranspose", "name": "upsample_9", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "upsample_9", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [2, 2]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 48, 96, 32]}}
°
regularization_losses
	variables
trainable_variables
	keras_api
Ζ__call__
+Η&call_and_return_all_conditional_losses"
_tf_keras_layer{"class_name": "TensorFlowOpLayer", "name": "tf_op_layer_concat_9", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "concat_9", "trainable": true, "dtype": "float32", "node_def": {"name": "concat_9", "op": "ConcatV2", "input": ["upsample_9/BiasAdd", "re_lu/Relu", "concat_9/axis"], "attr": {"Tidx": {"type": "DT_INT32"}, "T": {"type": "DT_FLOAT"}, "N": {"i": "2"}}}, "constants": {"2": -1}}}
ω	
kernel
	bias
regularization_losses
	variables
trainable_variables
	keras_api
Θ__call__
+Ι&call_and_return_all_conditional_losses"Μ
_tf_keras_layer²{"class_name": "Conv2D", "name": "conv_9", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv_9", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 96, 192, 32]}}
Ζ	
	axis

gamma
	beta
moving_mean
moving_variance
regularization_losses
	variables
trainable_variables
	keras_api
Κ__call__
+Λ&call_and_return_all_conditional_losses"η
_tf_keras_layerΝ{"class_name": "BatchNormalization", "name": "batch_normalization_8", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_8", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 16}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 96, 192, 16]}}
ρ
regularization_losses
	variables
trainable_variables
	keras_api
Μ__call__
+Ν&call_and_return_all_conditional_losses"ά
_tf_keras_layerΒ{"class_name": "ReLU", "name": "re_lu_8", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "re_lu_8", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}}
φ	
kernel
	bias
regularization_losses
	variables
 trainable_variables
‘	keras_api
Ξ__call__
+Ο&call_and_return_all_conditional_losses"Ι
_tf_keras_layer―{"class_name": "Conv2D", "name": "final", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "final", "trainable": true, "dtype": "float32", "filters": 2, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 16}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 96, 192, 16]}}
 "
trackable_list_wrapper
Β
90
:1
@2
A3
B4
C5
P6
Q7
W8
X9
Y10
Z11
g12
h13
n14
o15
p16
q17
~18
19
20
21
22
23
24
25
26
27
28
29
¨30
©31
²32
³33
Ή34
Ί35
»36
Ό37
Ε38
Ζ39
Ο40
Π41
Φ42
Χ43
Ψ44
Ω45
β46
γ47
μ48
ν49
σ50
τ51
υ52
φ53
?54
55
56
57
58
59
60
61
62
63"
trackable_list_wrapper
¦
90
:1
@2
A3
P4
Q5
W6
X7
g8
h9
n10
o11
~12
13
14
15
16
17
18
19
¨20
©21
²22
³23
Ή24
Ί25
Ε26
Ζ27
Ο28
Π29
Φ30
Χ31
β32
γ33
μ34
ν35
σ36
τ37
?38
39
40
41
42
43
44
45"
trackable_list_wrapper
Σ
,regularization_losses
’non_trainable_variables
-	variables
£metrics
 €layer_regularization_losses
₯layer_metrics
¦layers
.trainable_variables
ω__call__
ϋ_default_save_signature
+ϊ&call_and_return_all_conditional_losses
'ϊ"call_and_return_conditional_losses"
_generic_user_object
-
Πserving_default"
signature_map
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
΅
1regularization_losses
§non_trainable_variables
2	variables
¨metrics
 ©layer_regularization_losses
ͺlayer_metrics
«layers
3trainable_variables
ό__call__
+ύ&call_and_return_all_conditional_losses
'ύ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
΅
5regularization_losses
¬non_trainable_variables
6	variables
­metrics
 ?layer_regularization_losses
―layer_metrics
°layers
7trainable_variables
ώ__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
':%2conv_1/kernel
:2conv_1/bias
 "
trackable_list_wrapper
.
90
:1"
trackable_list_wrapper
.
90
:1"
trackable_list_wrapper
΅
;regularization_losses
±non_trainable_variables
<	variables
²metrics
 ³layer_regularization_losses
΄layer_metrics
΅layers
=trainable_variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
':%2batch_normalization/gamma
&:$2batch_normalization/beta
/:- (2batch_normalization/moving_mean
3:1 (2#batch_normalization/moving_variance
 "
trackable_list_wrapper
<
@0
A1
B2
C3"
trackable_list_wrapper
.
@0
A1"
trackable_list_wrapper
΅
Dregularization_losses
Άnon_trainable_variables
E	variables
·metrics
 Έlayer_regularization_losses
Ήlayer_metrics
Ίlayers
Ftrainable_variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
΅
Hregularization_losses
»non_trainable_variables
I	variables
Όmetrics
 ½layer_regularization_losses
Ύlayer_metrics
Ώlayers
Jtrainable_variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
΅
Lregularization_losses
ΐnon_trainable_variables
M	variables
Αmetrics
 Βlayer_regularization_losses
Γlayer_metrics
Δlayers
Ntrainable_variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
':% 2conv_2/kernel
: 2conv_2/bias
 "
trackable_list_wrapper
.
P0
Q1"
trackable_list_wrapper
.
P0
Q1"
trackable_list_wrapper
΅
Rregularization_losses
Εnon_trainable_variables
S	variables
Ζmetrics
 Ηlayer_regularization_losses
Θlayer_metrics
Ιlayers
Ttrainable_variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
):' 2batch_normalization_1/gamma
(:& 2batch_normalization_1/beta
1:/  (2!batch_normalization_1/moving_mean
5:3  (2%batch_normalization_1/moving_variance
 "
trackable_list_wrapper
<
W0
X1
Y2
Z3"
trackable_list_wrapper
.
W0
X1"
trackable_list_wrapper
΅
[regularization_losses
Κnon_trainable_variables
\	variables
Λmetrics
 Μlayer_regularization_losses
Νlayer_metrics
Ξlayers
]trainable_variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
΅
_regularization_losses
Οnon_trainable_variables
`	variables
Πmetrics
 Ρlayer_regularization_losses
?layer_metrics
Σlayers
atrainable_variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
΅
cregularization_losses
Τnon_trainable_variables
d	variables
Υmetrics
 Φlayer_regularization_losses
Χlayer_metrics
Ψlayers
etrainable_variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
':% @2conv_3/kernel
:@2conv_3/bias
 "
trackable_list_wrapper
.
g0
h1"
trackable_list_wrapper
.
g0
h1"
trackable_list_wrapper
΅
iregularization_losses
Ωnon_trainable_variables
j	variables
Ϊmetrics
 Ϋlayer_regularization_losses
άlayer_metrics
έlayers
ktrainable_variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
):'@2batch_normalization_2/gamma
(:&@2batch_normalization_2/beta
1:/@ (2!batch_normalization_2/moving_mean
5:3@ (2%batch_normalization_2/moving_variance
 "
trackable_list_wrapper
<
n0
o1
p2
q3"
trackable_list_wrapper
.
n0
o1"
trackable_list_wrapper
΅
rregularization_losses
ήnon_trainable_variables
s	variables
ίmetrics
 ΰlayer_regularization_losses
αlayer_metrics
βlayers
ttrainable_variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
΅
vregularization_losses
γnon_trainable_variables
w	variables
δmetrics
 εlayer_regularization_losses
ζlayer_metrics
ηlayers
xtrainable_variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
΅
zregularization_losses
θnon_trainable_variables
{	variables
ιmetrics
 κlayer_regularization_losses
λlayer_metrics
μlayers
|trainable_variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
(:&@2conv_4/kernel
:2conv_4/bias
 "
trackable_list_wrapper
.
~0
1"
trackable_list_wrapper
.
~0
1"
trackable_list_wrapper
Έ
regularization_losses
νnon_trainable_variables
	variables
ξmetrics
 οlayer_regularization_losses
πlayer_metrics
ρlayers
trainable_variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
*:(2batch_normalization_3/gamma
):'2batch_normalization_3/beta
2:0 (2!batch_normalization_3/moving_mean
6:4 (2%batch_normalization_3/moving_variance
 "
trackable_list_wrapper
@
0
1
2
3"
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
Έ
regularization_losses
ςnon_trainable_variables
	variables
σmetrics
 τlayer_regularization_losses
υlayer_metrics
φlayers
trainable_variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Έ
regularization_losses
χnon_trainable_variables
	variables
ψmetrics
 ωlayer_regularization_losses
ϊlayer_metrics
ϋlayers
trainable_variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Έ
regularization_losses
όnon_trainable_variables
	variables
ύmetrics
 ώlayer_regularization_losses
?layer_metrics
layers
trainable_variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
):'2conv_5/kernel
:2conv_5/bias
 "
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
Έ
regularization_losses
non_trainable_variables
	variables
metrics
 layer_regularization_losses
layer_metrics
layers
trainable_variables
 __call__
+‘&call_and_return_all_conditional_losses
'‘"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
*:(2batch_normalization_4/gamma
):'2batch_normalization_4/beta
2:0 (2!batch_normalization_4/moving_mean
6:4 (2%batch_normalization_4/moving_variance
 "
trackable_list_wrapper
@
0
1
2
3"
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
Έ
 regularization_losses
non_trainable_variables
‘	variables
metrics
 layer_regularization_losses
layer_metrics
layers
’trainable_variables
’__call__
+£&call_and_return_all_conditional_losses
'£"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Έ
€regularization_losses
non_trainable_variables
₯	variables
metrics
 layer_regularization_losses
layer_metrics
layers
¦trainable_variables
€__call__
+₯&call_and_return_all_conditional_losses
'₯"call_and_return_conditional_losses"
_generic_user_object
-:+2upsample_6/kernel
:2upsample_6/bias
 "
trackable_list_wrapper
0
¨0
©1"
trackable_list_wrapper
0
¨0
©1"
trackable_list_wrapper
Έ
ͺregularization_losses
non_trainable_variables
«	variables
metrics
 layer_regularization_losses
layer_metrics
layers
¬trainable_variables
¦__call__
+§&call_and_return_all_conditional_losses
'§"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Έ
?regularization_losses
non_trainable_variables
―	variables
metrics
 layer_regularization_losses
layer_metrics
layers
°trainable_variables
¨__call__
+©&call_and_return_all_conditional_losses
'©"call_and_return_conditional_losses"
_generic_user_object
):'2conv_6/kernel
:2conv_6/bias
 "
trackable_list_wrapper
0
²0
³1"
trackable_list_wrapper
0
²0
³1"
trackable_list_wrapper
Έ
΄regularization_losses
non_trainable_variables
΅	variables
metrics
 layer_regularization_losses
layer_metrics
layers
Άtrainable_variables
ͺ__call__
+«&call_and_return_all_conditional_losses
'«"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
*:(2batch_normalization_5/gamma
):'2batch_normalization_5/beta
2:0 (2!batch_normalization_5/moving_mean
6:4 (2%batch_normalization_5/moving_variance
 "
trackable_list_wrapper
@
Ή0
Ί1
»2
Ό3"
trackable_list_wrapper
0
Ή0
Ί1"
trackable_list_wrapper
Έ
½regularization_losses
non_trainable_variables
Ύ	variables
 metrics
 ‘layer_regularization_losses
’layer_metrics
£layers
Ώtrainable_variables
¬__call__
+­&call_and_return_all_conditional_losses
'­"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Έ
Αregularization_losses
€non_trainable_variables
Β	variables
₯metrics
 ¦layer_regularization_losses
§layer_metrics
¨layers
Γtrainable_variables
?__call__
+―&call_and_return_all_conditional_losses
'―"call_and_return_conditional_losses"
_generic_user_object
,:*@2upsample_7/kernel
:@2upsample_7/bias
 "
trackable_list_wrapper
0
Ε0
Ζ1"
trackable_list_wrapper
0
Ε0
Ζ1"
trackable_list_wrapper
Έ
Ηregularization_losses
©non_trainable_variables
Θ	variables
ͺmetrics
 «layer_regularization_losses
¬layer_metrics
­layers
Ιtrainable_variables
°__call__
+±&call_and_return_all_conditional_losses
'±"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Έ
Λregularization_losses
?non_trainable_variables
Μ	variables
―metrics
 °layer_regularization_losses
±layer_metrics
²layers
Νtrainable_variables
²__call__
+³&call_and_return_all_conditional_losses
'³"call_and_return_conditional_losses"
_generic_user_object
(:&@2conv_7/kernel
:@2conv_7/bias
 "
trackable_list_wrapper
0
Ο0
Π1"
trackable_list_wrapper
0
Ο0
Π1"
trackable_list_wrapper
Έ
Ρregularization_losses
³non_trainable_variables
?	variables
΄metrics
 ΅layer_regularization_losses
Άlayer_metrics
·layers
Σtrainable_variables
΄__call__
+΅&call_and_return_all_conditional_losses
'΅"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
):'@2batch_normalization_6/gamma
(:&@2batch_normalization_6/beta
1:/@ (2!batch_normalization_6/moving_mean
5:3@ (2%batch_normalization_6/moving_variance
 "
trackable_list_wrapper
@
Φ0
Χ1
Ψ2
Ω3"
trackable_list_wrapper
0
Φ0
Χ1"
trackable_list_wrapper
Έ
Ϊregularization_losses
Έnon_trainable_variables
Ϋ	variables
Ήmetrics
 Ίlayer_regularization_losses
»layer_metrics
Όlayers
άtrainable_variables
Ά__call__
+·&call_and_return_all_conditional_losses
'·"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Έ
ήregularization_losses
½non_trainable_variables
ί	variables
Ύmetrics
 Ώlayer_regularization_losses
ΐlayer_metrics
Αlayers
ΰtrainable_variables
Έ__call__
+Ή&call_and_return_all_conditional_losses
'Ή"call_and_return_conditional_losses"
_generic_user_object
+:) @2upsample_8/kernel
: 2upsample_8/bias
 "
trackable_list_wrapper
0
β0
γ1"
trackable_list_wrapper
0
β0
γ1"
trackable_list_wrapper
Έ
δregularization_losses
Βnon_trainable_variables
ε	variables
Γmetrics
 Δlayer_regularization_losses
Εlayer_metrics
Ζlayers
ζtrainable_variables
Ί__call__
+»&call_and_return_all_conditional_losses
'»"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Έ
θregularization_losses
Ηnon_trainable_variables
ι	variables
Θmetrics
 Ιlayer_regularization_losses
Κlayer_metrics
Λlayers
κtrainable_variables
Ό__call__
+½&call_and_return_all_conditional_losses
'½"call_and_return_conditional_losses"
_generic_user_object
':%@ 2conv_8/kernel
: 2conv_8/bias
 "
trackable_list_wrapper
0
μ0
ν1"
trackable_list_wrapper
0
μ0
ν1"
trackable_list_wrapper
Έ
ξregularization_losses
Μnon_trainable_variables
ο	variables
Νmetrics
 Ξlayer_regularization_losses
Οlayer_metrics
Πlayers
πtrainable_variables
Ύ__call__
+Ώ&call_and_return_all_conditional_losses
'Ώ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
):' 2batch_normalization_7/gamma
(:& 2batch_normalization_7/beta
1:/  (2!batch_normalization_7/moving_mean
5:3  (2%batch_normalization_7/moving_variance
 "
trackable_list_wrapper
@
σ0
τ1
υ2
φ3"
trackable_list_wrapper
0
σ0
τ1"
trackable_list_wrapper
Έ
χregularization_losses
Ρnon_trainable_variables
ψ	variables
?metrics
 Σlayer_regularization_losses
Τlayer_metrics
Υlayers
ωtrainable_variables
ΐ__call__
+Α&call_and_return_all_conditional_losses
'Α"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Έ
ϋregularization_losses
Φnon_trainable_variables
ό	variables
Χmetrics
 Ψlayer_regularization_losses
Ωlayer_metrics
Ϊlayers
ύtrainable_variables
Β__call__
+Γ&call_and_return_all_conditional_losses
'Γ"call_and_return_conditional_losses"
_generic_user_object
+:) 2upsample_9/kernel
:2upsample_9/bias
 "
trackable_list_wrapper
0
?0
1"
trackable_list_wrapper
0
?0
1"
trackable_list_wrapper
Έ
regularization_losses
Ϋnon_trainable_variables
	variables
άmetrics
 έlayer_regularization_losses
ήlayer_metrics
ίlayers
trainable_variables
Δ__call__
+Ε&call_and_return_all_conditional_losses
'Ε"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Έ
regularization_losses
ΰnon_trainable_variables
	variables
αmetrics
 βlayer_regularization_losses
γlayer_metrics
δlayers
trainable_variables
Ζ__call__
+Η&call_and_return_all_conditional_losses
'Η"call_and_return_conditional_losses"
_generic_user_object
':% 2conv_9/kernel
:2conv_9/bias
 "
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
Έ
regularization_losses
εnon_trainable_variables
	variables
ζmetrics
 ηlayer_regularization_losses
θlayer_metrics
ιlayers
trainable_variables
Θ__call__
+Ι&call_and_return_all_conditional_losses
'Ι"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
):'2batch_normalization_8/gamma
(:&2batch_normalization_8/beta
1:/ (2!batch_normalization_8/moving_mean
5:3 (2%batch_normalization_8/moving_variance
 "
trackable_list_wrapper
@
0
1
2
3"
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
Έ
regularization_losses
κnon_trainable_variables
	variables
λmetrics
 μlayer_regularization_losses
νlayer_metrics
ξlayers
trainable_variables
Κ__call__
+Λ&call_and_return_all_conditional_losses
'Λ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Έ
regularization_losses
οnon_trainable_variables
	variables
πmetrics
 ρlayer_regularization_losses
ςlayer_metrics
σlayers
trainable_variables
Μ__call__
+Ν&call_and_return_all_conditional_losses
'Ν"call_and_return_conditional_losses"
_generic_user_object
&:$2final/kernel
:2
final/bias
 "
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
Έ
regularization_losses
τnon_trainable_variables
	variables
υmetrics
 φlayer_regularization_losses
χlayer_metrics
ψlayers
 trainable_variables
Ξ__call__
+Ο&call_and_return_all_conditional_losses
'Ο"call_and_return_conditional_losses"
_generic_user_object
²
B0
C1
Y2
Z3
p4
q5
6
7
8
9
»10
Ό11
Ψ12
Ω13
υ14
φ15
16
17"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ξ
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28
29
30
 31
!32
"33
#34
$35
%36
&37
'38
(39
)40
*41
+42"
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
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
B0
C1"
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
.
Y0
Z1"
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
.
p0
q1"
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
0
0
1"
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
0
0
1"
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
0
»0
Ό1"
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
0
Ψ0
Ω1"
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
0
υ0
φ1"
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
0
0
1"
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
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
2
/__inference_functional_1_layer_call_fn_57381067
/__inference_functional_1_layer_call_fn_57380199
/__inference_functional_1_layer_call_fn_57381200
/__inference_functional_1_layer_call_fn_57379892ΐ
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
φ2σ
J__inference_functional_1_layer_call_and_return_conditional_losses_57379584
J__inference_functional_1_layer_call_and_return_conditional_losses_57380643
J__inference_functional_1_layer_call_and_return_conditional_losses_57379410
J__inference_functional_1_layer_call_and_return_conditional_losses_57380934ΐ
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
ν2κ
#__inference__wrapped_model_57377079Β
²
FullArgSpec
args 
varargsjargs
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *2’/
-*

imageInput?????????`ΐ
ΰ2έ
6__inference_tf_op_layer_RealDiv_layer_call_fn_57381211’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
ϋ2ψ
Q__inference_tf_op_layer_RealDiv_layer_call_and_return_conditional_losses_57381206’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
ά2Ω
2__inference_tf_op_layer_Sub_layer_call_fn_57381222’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
χ2τ
M__inference_tf_op_layer_Sub_layer_call_and_return_conditional_losses_57381217’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
Σ2Π
)__inference_conv_1_layer_call_fn_57381241’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
ξ2λ
D__inference_conv_1_layer_call_and_return_conditional_losses_57381232’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
2
6__inference_batch_normalization_layer_call_fn_57381356
6__inference_batch_normalization_layer_call_fn_57381305
6__inference_batch_normalization_layer_call_fn_57381369
6__inference_batch_normalization_layer_call_fn_57381292΄
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
2
Q__inference_batch_normalization_layer_call_and_return_conditional_losses_57381343
Q__inference_batch_normalization_layer_call_and_return_conditional_losses_57381261
Q__inference_batch_normalization_layer_call_and_return_conditional_losses_57381279
Q__inference_batch_normalization_layer_call_and_return_conditional_losses_57381325΄
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
?2Ο
(__inference_re_lu_layer_call_fn_57381379’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
ν2κ
C__inference_re_lu_layer_call_and_return_conditional_losses_57381374’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
2
0__inference_max_pooling2d_layer_call_fn_57377195ΰ
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *@’=
;84????????????????????????????????????
³2°
K__inference_max_pooling2d_layer_call_and_return_conditional_losses_57377189ΰ
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *@’=
;84????????????????????????????????????
Σ2Π
)__inference_conv_2_layer_call_fn_57381398’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
ξ2λ
D__inference_conv_2_layer_call_and_return_conditional_losses_57381389’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
’2
8__inference_batch_normalization_1_layer_call_fn_57381513
8__inference_batch_normalization_1_layer_call_fn_57381462
8__inference_batch_normalization_1_layer_call_fn_57381526
8__inference_batch_normalization_1_layer_call_fn_57381449΄
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
2
S__inference_batch_normalization_1_layer_call_and_return_conditional_losses_57381436
S__inference_batch_normalization_1_layer_call_and_return_conditional_losses_57381418
S__inference_batch_normalization_1_layer_call_and_return_conditional_losses_57381500
S__inference_batch_normalization_1_layer_call_and_return_conditional_losses_57381482΄
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
Τ2Ρ
*__inference_re_lu_1_layer_call_fn_57381536’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
ο2μ
E__inference_re_lu_1_layer_call_and_return_conditional_losses_57381531’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
2
2__inference_max_pooling2d_1_layer_call_fn_57377311ΰ
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *@’=
;84????????????????????????????????????
΅2²
M__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_57377305ΰ
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *@’=
;84????????????????????????????????????
Σ2Π
)__inference_conv_3_layer_call_fn_57381555’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
ξ2λ
D__inference_conv_3_layer_call_and_return_conditional_losses_57381546’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
’2
8__inference_batch_normalization_2_layer_call_fn_57381683
8__inference_batch_normalization_2_layer_call_fn_57381619
8__inference_batch_normalization_2_layer_call_fn_57381606
8__inference_batch_normalization_2_layer_call_fn_57381670΄
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
2
S__inference_batch_normalization_2_layer_call_and_return_conditional_losses_57381657
S__inference_batch_normalization_2_layer_call_and_return_conditional_losses_57381639
S__inference_batch_normalization_2_layer_call_and_return_conditional_losses_57381593
S__inference_batch_normalization_2_layer_call_and_return_conditional_losses_57381575΄
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
Τ2Ρ
*__inference_re_lu_2_layer_call_fn_57381693’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
ο2μ
E__inference_re_lu_2_layer_call_and_return_conditional_losses_57381688’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
2
2__inference_max_pooling2d_2_layer_call_fn_57377427ΰ
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *@’=
;84????????????????????????????????????
΅2²
M__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_57377421ΰ
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *@’=
;84????????????????????????????????????
Σ2Π
)__inference_conv_4_layer_call_fn_57381712’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
ξ2λ
D__inference_conv_4_layer_call_and_return_conditional_losses_57381703’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
’2
8__inference_batch_normalization_3_layer_call_fn_57381776
8__inference_batch_normalization_3_layer_call_fn_57381827
8__inference_batch_normalization_3_layer_call_fn_57381763
8__inference_batch_normalization_3_layer_call_fn_57381840΄
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
2
S__inference_batch_normalization_3_layer_call_and_return_conditional_losses_57381750
S__inference_batch_normalization_3_layer_call_and_return_conditional_losses_57381796
S__inference_batch_normalization_3_layer_call_and_return_conditional_losses_57381732
S__inference_batch_normalization_3_layer_call_and_return_conditional_losses_57381814΄
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
Τ2Ρ
*__inference_re_lu_3_layer_call_fn_57381850’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
ο2μ
E__inference_re_lu_3_layer_call_and_return_conditional_losses_57381845’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
2
2__inference_max_pooling2d_3_layer_call_fn_57377543ΰ
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *@’=
;84????????????????????????????????????
΅2²
M__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_57377537ΰ
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *@’=
;84????????????????????????????????????
Σ2Π
)__inference_conv_5_layer_call_fn_57381869’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
ξ2λ
D__inference_conv_5_layer_call_and_return_conditional_losses_57381860’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
’2
8__inference_batch_normalization_4_layer_call_fn_57381997
8__inference_batch_normalization_4_layer_call_fn_57381984
8__inference_batch_normalization_4_layer_call_fn_57381920
8__inference_batch_normalization_4_layer_call_fn_57381933΄
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
2
S__inference_batch_normalization_4_layer_call_and_return_conditional_losses_57381953
S__inference_batch_normalization_4_layer_call_and_return_conditional_losses_57381971
S__inference_batch_normalization_4_layer_call_and_return_conditional_losses_57381907
S__inference_batch_normalization_4_layer_call_and_return_conditional_losses_57381889΄
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
Τ2Ρ
*__inference_re_lu_4_layer_call_fn_57382007’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
ο2μ
E__inference_re_lu_4_layer_call_and_return_conditional_losses_57382002’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
2
-__inference_upsample_6_layer_call_fn_57377695Ψ
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *8’5
30,???????????????????????????
¨2₯
H__inference_upsample_6_layer_call_and_return_conditional_losses_57377685Ψ
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *8’5
30,???????????????????????????
α2ή
7__inference_tf_op_layer_concat_6_layer_call_fn_57382020’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
ό2ω
R__inference_tf_op_layer_concat_6_layer_call_and_return_conditional_losses_57382014’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
Σ2Π
)__inference_conv_6_layer_call_fn_57382039’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
ξ2λ
D__inference_conv_6_layer_call_and_return_conditional_losses_57382030’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
’2
8__inference_batch_normalization_5_layer_call_fn_57382167
8__inference_batch_normalization_5_layer_call_fn_57382154
8__inference_batch_normalization_5_layer_call_fn_57382090
8__inference_batch_normalization_5_layer_call_fn_57382103΄
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
2
S__inference_batch_normalization_5_layer_call_and_return_conditional_losses_57382141
S__inference_batch_normalization_5_layer_call_and_return_conditional_losses_57382077
S__inference_batch_normalization_5_layer_call_and_return_conditional_losses_57382123
S__inference_batch_normalization_5_layer_call_and_return_conditional_losses_57382059΄
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
Τ2Ρ
*__inference_re_lu_5_layer_call_fn_57382177’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
ο2μ
E__inference_re_lu_5_layer_call_and_return_conditional_losses_57382172’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
2
-__inference_upsample_7_layer_call_fn_57377847Ψ
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *8’5
30,???????????????????????????
¨2₯
H__inference_upsample_7_layer_call_and_return_conditional_losses_57377837Ψ
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *8’5
30,???????????????????????????
α2ή
7__inference_tf_op_layer_concat_7_layer_call_fn_57382190’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
ό2ω
R__inference_tf_op_layer_concat_7_layer_call_and_return_conditional_losses_57382184’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
Σ2Π
)__inference_conv_7_layer_call_fn_57382209’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
ξ2λ
D__inference_conv_7_layer_call_and_return_conditional_losses_57382200’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
’2
8__inference_batch_normalization_6_layer_call_fn_57382273
8__inference_batch_normalization_6_layer_call_fn_57382260
8__inference_batch_normalization_6_layer_call_fn_57382324
8__inference_batch_normalization_6_layer_call_fn_57382337΄
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
2
S__inference_batch_normalization_6_layer_call_and_return_conditional_losses_57382293
S__inference_batch_normalization_6_layer_call_and_return_conditional_losses_57382229
S__inference_batch_normalization_6_layer_call_and_return_conditional_losses_57382311
S__inference_batch_normalization_6_layer_call_and_return_conditional_losses_57382247΄
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
Τ2Ρ
*__inference_re_lu_6_layer_call_fn_57382347’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
ο2μ
E__inference_re_lu_6_layer_call_and_return_conditional_losses_57382342’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
2
-__inference_upsample_8_layer_call_fn_57377999Χ
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *7’4
2/+???????????????????????????@
§2€
H__inference_upsample_8_layer_call_and_return_conditional_losses_57377989Χ
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *7’4
2/+???????????????????????????@
α2ή
7__inference_tf_op_layer_concat_8_layer_call_fn_57382360’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
ό2ω
R__inference_tf_op_layer_concat_8_layer_call_and_return_conditional_losses_57382354’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
Σ2Π
)__inference_conv_8_layer_call_fn_57382379’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
ξ2λ
D__inference_conv_8_layer_call_and_return_conditional_losses_57382370’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
’2
8__inference_batch_normalization_7_layer_call_fn_57382443
8__inference_batch_normalization_7_layer_call_fn_57382430
8__inference_batch_normalization_7_layer_call_fn_57382507
8__inference_batch_normalization_7_layer_call_fn_57382494΄
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
2
S__inference_batch_normalization_7_layer_call_and_return_conditional_losses_57382463
S__inference_batch_normalization_7_layer_call_and_return_conditional_losses_57382399
S__inference_batch_normalization_7_layer_call_and_return_conditional_losses_57382417
S__inference_batch_normalization_7_layer_call_and_return_conditional_losses_57382481΄
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
Τ2Ρ
*__inference_re_lu_7_layer_call_fn_57382517’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
ο2μ
E__inference_re_lu_7_layer_call_and_return_conditional_losses_57382512’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
2
-__inference_upsample_9_layer_call_fn_57378151Χ
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *7’4
2/+??????????????????????????? 
§2€
H__inference_upsample_9_layer_call_and_return_conditional_losses_57378141Χ
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *7’4
2/+??????????????????????????? 
α2ή
7__inference_tf_op_layer_concat_9_layer_call_fn_57382530’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
ό2ω
R__inference_tf_op_layer_concat_9_layer_call_and_return_conditional_losses_57382524’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
Σ2Π
)__inference_conv_9_layer_call_fn_57382549’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
ξ2λ
D__inference_conv_9_layer_call_and_return_conditional_losses_57382540’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
’2
8__inference_batch_normalization_8_layer_call_fn_57382613
8__inference_batch_normalization_8_layer_call_fn_57382664
8__inference_batch_normalization_8_layer_call_fn_57382600
8__inference_batch_normalization_8_layer_call_fn_57382677΄
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
2
S__inference_batch_normalization_8_layer_call_and_return_conditional_losses_57382569
S__inference_batch_normalization_8_layer_call_and_return_conditional_losses_57382633
S__inference_batch_normalization_8_layer_call_and_return_conditional_losses_57382587
S__inference_batch_normalization_8_layer_call_and_return_conditional_losses_57382651΄
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
Τ2Ρ
*__inference_re_lu_8_layer_call_fn_57382687’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
ο2μ
E__inference_re_lu_8_layer_call_and_return_conditional_losses_57382682’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
?2Ο
(__inference_final_layer_call_fn_57382706’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
ν2κ
C__inference_final_layer_call_and_return_conditional_losses_57382697’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
8B6
&__inference_signature_wrapper_57380334
imageInput
#__inference__wrapped_model_57377079δl9:@ABCPQWXYZghnopq~¨©²³ΉΊ»ΌΕΖΟΠΦΧΨΩβγμνστυφ?<’9
2’/
-*

imageInput?????????`ΐ
ͺ "6ͺ3
1
final(%
final?????????`ΐξ
S__inference_batch_normalization_1_layer_call_and_return_conditional_losses_57381418WXYZM’J
C’@
:7
inputs+??????????????????????????? 
p
ͺ "?’<
52
0+??????????????????????????? 
 ξ
S__inference_batch_normalization_1_layer_call_and_return_conditional_losses_57381436WXYZM’J
C’@
:7
inputs+??????????????????????????? 
p 
ͺ "?’<
52
0+??????????????????????????? 
 Ι
S__inference_batch_normalization_1_layer_call_and_return_conditional_losses_57381482rWXYZ;’8
1’.
(%
inputs?????????0` 
p
ͺ "-’*
# 
0?????????0` 
 Ι
S__inference_batch_normalization_1_layer_call_and_return_conditional_losses_57381500rWXYZ;’8
1’.
(%
inputs?????????0` 
p 
ͺ "-’*
# 
0?????????0` 
 Ζ
8__inference_batch_normalization_1_layer_call_fn_57381449WXYZM’J
C’@
:7
inputs+??????????????????????????? 
p
ͺ "2/+??????????????????????????? Ζ
8__inference_batch_normalization_1_layer_call_fn_57381462WXYZM’J
C’@
:7
inputs+??????????????????????????? 
p 
ͺ "2/+??????????????????????????? ‘
8__inference_batch_normalization_1_layer_call_fn_57381513eWXYZ;’8
1’.
(%
inputs?????????0` 
p
ͺ " ?????????0` ‘
8__inference_batch_normalization_1_layer_call_fn_57381526eWXYZ;’8
1’.
(%
inputs?????????0` 
p 
ͺ " ?????????0` Ι
S__inference_batch_normalization_2_layer_call_and_return_conditional_losses_57381575rnopq;’8
1’.
(%
inputs?????????0@
p
ͺ "-’*
# 
0?????????0@
 Ι
S__inference_batch_normalization_2_layer_call_and_return_conditional_losses_57381593rnopq;’8
1’.
(%
inputs?????????0@
p 
ͺ "-’*
# 
0?????????0@
 ξ
S__inference_batch_normalization_2_layer_call_and_return_conditional_losses_57381639nopqM’J
C’@
:7
inputs+???????????????????????????@
p
ͺ "?’<
52
0+???????????????????????????@
 ξ
S__inference_batch_normalization_2_layer_call_and_return_conditional_losses_57381657nopqM’J
C’@
:7
inputs+???????????????????????????@
p 
ͺ "?’<
52
0+???????????????????????????@
 ‘
8__inference_batch_normalization_2_layer_call_fn_57381606enopq;’8
1’.
(%
inputs?????????0@
p
ͺ " ?????????0@‘
8__inference_batch_normalization_2_layer_call_fn_57381619enopq;’8
1’.
(%
inputs?????????0@
p 
ͺ " ?????????0@Ζ
8__inference_batch_normalization_2_layer_call_fn_57381670nopqM’J
C’@
:7
inputs+???????????????????????????@
p
ͺ "2/+???????????????????????????@Ζ
8__inference_batch_normalization_2_layer_call_fn_57381683nopqM’J
C’@
:7
inputs+???????????????????????????@
p 
ͺ "2/+???????????????????????????@τ
S__inference_batch_normalization_3_layer_call_and_return_conditional_losses_57381732N’K
D’A
;8
inputs,???????????????????????????
p
ͺ "@’=
63
0,???????????????????????????
 τ
S__inference_batch_normalization_3_layer_call_and_return_conditional_losses_57381750N’K
D’A
;8
inputs,???????????????????????????
p 
ͺ "@’=
63
0,???????????????????????????
 Ο
S__inference_batch_normalization_3_layer_call_and_return_conditional_losses_57381796x<’9
2’/
)&
inputs?????????
p
ͺ ".’+
$!
0?????????
 Ο
S__inference_batch_normalization_3_layer_call_and_return_conditional_losses_57381814x<’9
2’/
)&
inputs?????????
p 
ͺ ".’+
$!
0?????????
 Μ
8__inference_batch_normalization_3_layer_call_fn_57381763N’K
D’A
;8
inputs,???????????????????????????
p
ͺ "30,???????????????????????????Μ
8__inference_batch_normalization_3_layer_call_fn_57381776N’K
D’A
;8
inputs,???????????????????????????
p 
ͺ "30,???????????????????????????§
8__inference_batch_normalization_3_layer_call_fn_57381827k<’9
2’/
)&
inputs?????????
p
ͺ "!?????????§
8__inference_batch_normalization_3_layer_call_fn_57381840k<’9
2’/
)&
inputs?????????
p 
ͺ "!?????????Ο
S__inference_batch_normalization_4_layer_call_and_return_conditional_losses_57381889x<’9
2’/
)&
inputs?????????
p
ͺ ".’+
$!
0?????????
 Ο
S__inference_batch_normalization_4_layer_call_and_return_conditional_losses_57381907x<’9
2’/
)&
inputs?????????
p 
ͺ ".’+
$!
0?????????
 τ
S__inference_batch_normalization_4_layer_call_and_return_conditional_losses_57381953N’K
D’A
;8
inputs,???????????????????????????
p
ͺ "@’=
63
0,???????????????????????????
 τ
S__inference_batch_normalization_4_layer_call_and_return_conditional_losses_57381971N’K
D’A
;8
inputs,???????????????????????????
p 
ͺ "@’=
63
0,???????????????????????????
 §
8__inference_batch_normalization_4_layer_call_fn_57381920k<’9
2’/
)&
inputs?????????
p
ͺ "!?????????§
8__inference_batch_normalization_4_layer_call_fn_57381933k<’9
2’/
)&
inputs?????????
p 
ͺ "!?????????Μ
8__inference_batch_normalization_4_layer_call_fn_57381984N’K
D’A
;8
inputs,???????????????????????????
p
ͺ "30,???????????????????????????Μ
8__inference_batch_normalization_4_layer_call_fn_57381997N’K
D’A
;8
inputs,???????????????????????????
p 
ͺ "30,???????????????????????????Ο
S__inference_batch_normalization_5_layer_call_and_return_conditional_losses_57382059xΉΊ»Ό<’9
2’/
)&
inputs?????????
p
ͺ ".’+
$!
0?????????
 Ο
S__inference_batch_normalization_5_layer_call_and_return_conditional_losses_57382077xΉΊ»Ό<’9
2’/
)&
inputs?????????
p 
ͺ ".’+
$!
0?????????
 τ
S__inference_batch_normalization_5_layer_call_and_return_conditional_losses_57382123ΉΊ»ΌN’K
D’A
;8
inputs,???????????????????????????
p
ͺ "@’=
63
0,???????????????????????????
 τ
S__inference_batch_normalization_5_layer_call_and_return_conditional_losses_57382141ΉΊ»ΌN’K
D’A
;8
inputs,???????????????????????????
p 
ͺ "@’=
63
0,???????????????????????????
 §
8__inference_batch_normalization_5_layer_call_fn_57382090kΉΊ»Ό<’9
2’/
)&
inputs?????????
p
ͺ "!?????????§
8__inference_batch_normalization_5_layer_call_fn_57382103kΉΊ»Ό<’9
2’/
)&
inputs?????????
p 
ͺ "!?????????Μ
8__inference_batch_normalization_5_layer_call_fn_57382154ΉΊ»ΌN’K
D’A
;8
inputs,???????????????????????????
p
ͺ "30,???????????????????????????Μ
8__inference_batch_normalization_5_layer_call_fn_57382167ΉΊ»ΌN’K
D’A
;8
inputs,???????????????????????????
p 
ͺ "30,???????????????????????????Ν
S__inference_batch_normalization_6_layer_call_and_return_conditional_losses_57382229vΦΧΨΩ;’8
1’.
(%
inputs?????????0@
p
ͺ "-’*
# 
0?????????0@
 Ν
S__inference_batch_normalization_6_layer_call_and_return_conditional_losses_57382247vΦΧΨΩ;’8
1’.
(%
inputs?????????0@
p 
ͺ "-’*
# 
0?????????0@
 ς
S__inference_batch_normalization_6_layer_call_and_return_conditional_losses_57382293ΦΧΨΩM’J
C’@
:7
inputs+???????????????????????????@
p
ͺ "?’<
52
0+???????????????????????????@
 ς
S__inference_batch_normalization_6_layer_call_and_return_conditional_losses_57382311ΦΧΨΩM’J
C’@
:7
inputs+???????????????????????????@
p 
ͺ "?’<
52
0+???????????????????????????@
 ₯
8__inference_batch_normalization_6_layer_call_fn_57382260iΦΧΨΩ;’8
1’.
(%
inputs?????????0@
p
ͺ " ?????????0@₯
8__inference_batch_normalization_6_layer_call_fn_57382273iΦΧΨΩ;’8
1’.
(%
inputs?????????0@
p 
ͺ " ?????????0@Κ
8__inference_batch_normalization_6_layer_call_fn_57382324ΦΧΨΩM’J
C’@
:7
inputs+???????????????????????????@
p
ͺ "2/+???????????????????????????@Κ
8__inference_batch_normalization_6_layer_call_fn_57382337ΦΧΨΩM’J
C’@
:7
inputs+???????????????????????????@
p 
ͺ "2/+???????????????????????????@ς
S__inference_batch_normalization_7_layer_call_and_return_conditional_losses_57382399στυφM’J
C’@
:7
inputs+??????????????????????????? 
p
ͺ "?’<
52
0+??????????????????????????? 
 ς
S__inference_batch_normalization_7_layer_call_and_return_conditional_losses_57382417στυφM’J
C’@
:7
inputs+??????????????????????????? 
p 
ͺ "?’<
52
0+??????????????????????????? 
 Ν
S__inference_batch_normalization_7_layer_call_and_return_conditional_losses_57382463vστυφ;’8
1’.
(%
inputs?????????0` 
p
ͺ "-’*
# 
0?????????0` 
 Ν
S__inference_batch_normalization_7_layer_call_and_return_conditional_losses_57382481vστυφ;’8
1’.
(%
inputs?????????0` 
p 
ͺ "-’*
# 
0?????????0` 
 Κ
8__inference_batch_normalization_7_layer_call_fn_57382430στυφM’J
C’@
:7
inputs+??????????????????????????? 
p
ͺ "2/+??????????????????????????? Κ
8__inference_batch_normalization_7_layer_call_fn_57382443στυφM’J
C’@
:7
inputs+??????????????????????????? 
p 
ͺ "2/+??????????????????????????? ₯
8__inference_batch_normalization_7_layer_call_fn_57382494iστυφ;’8
1’.
(%
inputs?????????0` 
p
ͺ " ?????????0` ₯
8__inference_batch_normalization_7_layer_call_fn_57382507iστυφ;’8
1’.
(%
inputs?????????0` 
p 
ͺ " ?????????0` ς
S__inference_batch_normalization_8_layer_call_and_return_conditional_losses_57382569M’J
C’@
:7
inputs+???????????????????????????
p
ͺ "?’<
52
0+???????????????????????????
 ς
S__inference_batch_normalization_8_layer_call_and_return_conditional_losses_57382587M’J
C’@
:7
inputs+???????????????????????????
p 
ͺ "?’<
52
0+???????????????????????????
 Ο
S__inference_batch_normalization_8_layer_call_and_return_conditional_losses_57382633x<’9
2’/
)&
inputs?????????`ΐ
p
ͺ ".’+
$!
0?????????`ΐ
 Ο
S__inference_batch_normalization_8_layer_call_and_return_conditional_losses_57382651x<’9
2’/
)&
inputs?????????`ΐ
p 
ͺ ".’+
$!
0?????????`ΐ
 Κ
8__inference_batch_normalization_8_layer_call_fn_57382600M’J
C’@
:7
inputs+???????????????????????????
p
ͺ "2/+???????????????????????????Κ
8__inference_batch_normalization_8_layer_call_fn_57382613M’J
C’@
:7
inputs+???????????????????????????
p 
ͺ "2/+???????????????????????????§
8__inference_batch_normalization_8_layer_call_fn_57382664k<’9
2’/
)&
inputs?????????`ΐ
p
ͺ "!?????????`ΐ§
8__inference_batch_normalization_8_layer_call_fn_57382677k<’9
2’/
)&
inputs?????????`ΐ
p 
ͺ "!?????????`ΐμ
Q__inference_batch_normalization_layer_call_and_return_conditional_losses_57381261@ABCM’J
C’@
:7
inputs+???????????????????????????
p
ͺ "?’<
52
0+???????????????????????????
 μ
Q__inference_batch_normalization_layer_call_and_return_conditional_losses_57381279@ABCM’J
C’@
:7
inputs+???????????????????????????
p 
ͺ "?’<
52
0+???????????????????????????
 Ι
Q__inference_batch_normalization_layer_call_and_return_conditional_losses_57381325t@ABC<’9
2’/
)&
inputs?????????`ΐ
p
ͺ ".’+
$!
0?????????`ΐ
 Ι
Q__inference_batch_normalization_layer_call_and_return_conditional_losses_57381343t@ABC<’9
2’/
)&
inputs?????????`ΐ
p 
ͺ ".’+
$!
0?????????`ΐ
 Δ
6__inference_batch_normalization_layer_call_fn_57381292@ABCM’J
C’@
:7
inputs+???????????????????????????
p
ͺ "2/+???????????????????????????Δ
6__inference_batch_normalization_layer_call_fn_57381305@ABCM’J
C’@
:7
inputs+???????????????????????????
p 
ͺ "2/+???????????????????????????‘
6__inference_batch_normalization_layer_call_fn_57381356g@ABC<’9
2’/
)&
inputs?????????`ΐ
p
ͺ "!?????????`ΐ‘
6__inference_batch_normalization_layer_call_fn_57381369g@ABC<’9
2’/
)&
inputs?????????`ΐ
p 
ͺ "!?????????`ΐΆ
D__inference_conv_1_layer_call_and_return_conditional_losses_57381232n9:8’5
.’+
)&
inputs?????????`ΐ
ͺ ".’+
$!
0?????????`ΐ
 
)__inference_conv_1_layer_call_fn_57381241a9:8’5
.’+
)&
inputs?????????`ΐ
ͺ "!?????????`ΐ΄
D__inference_conv_2_layer_call_and_return_conditional_losses_57381389lPQ7’4
-’*
(%
inputs?????????0`
ͺ "-’*
# 
0?????????0` 
 
)__inference_conv_2_layer_call_fn_57381398_PQ7’4
-’*
(%
inputs?????????0`
ͺ " ?????????0` ΄
D__inference_conv_3_layer_call_and_return_conditional_losses_57381546lgh7’4
-’*
(%
inputs?????????0 
ͺ "-’*
# 
0?????????0@
 
)__inference_conv_3_layer_call_fn_57381555_gh7’4
-’*
(%
inputs?????????0 
ͺ " ?????????0@΅
D__inference_conv_4_layer_call_and_return_conditional_losses_57381703m~7’4
-’*
(%
inputs?????????@
ͺ ".’+
$!
0?????????
 
)__inference_conv_4_layer_call_fn_57381712`~7’4
-’*
(%
inputs?????????@
ͺ "!?????????Έ
D__inference_conv_5_layer_call_and_return_conditional_losses_57381860p8’5
.’+
)&
inputs?????????
ͺ ".’+
$!
0?????????
 
)__inference_conv_5_layer_call_fn_57381869c8’5
.’+
)&
inputs?????????
ͺ "!?????????Έ
D__inference_conv_6_layer_call_and_return_conditional_losses_57382030p²³8’5
.’+
)&
inputs?????????
ͺ ".’+
$!
0?????????
 
)__inference_conv_6_layer_call_fn_57382039c²³8’5
.’+
)&
inputs?????????
ͺ "!?????????·
D__inference_conv_7_layer_call_and_return_conditional_losses_57382200oΟΠ8’5
.’+
)&
inputs?????????0
ͺ "-’*
# 
0?????????0@
 
)__inference_conv_7_layer_call_fn_57382209bΟΠ8’5
.’+
)&
inputs?????????0
ͺ " ?????????0@Ά
D__inference_conv_8_layer_call_and_return_conditional_losses_57382370nμν7’4
-’*
(%
inputs?????????0`@
ͺ "-’*
# 
0?????????0` 
 
)__inference_conv_8_layer_call_fn_57382379aμν7’4
-’*
(%
inputs?????????0`@
ͺ " ?????????0` Έ
D__inference_conv_9_layer_call_and_return_conditional_losses_57382540p8’5
.’+
)&
inputs?????????`ΐ 
ͺ ".’+
$!
0?????????`ΐ
 
)__inference_conv_9_layer_call_fn_57382549c8’5
.’+
)&
inputs?????????`ΐ 
ͺ "!?????????`ΐ·
C__inference_final_layer_call_and_return_conditional_losses_57382697p8’5
.’+
)&
inputs?????????`ΐ
ͺ ".’+
$!
0?????????`ΐ
 
(__inference_final_layer_call_fn_57382706c8’5
.’+
)&
inputs?????????`ΐ
ͺ "!?????????`ΐ³
J__inference_functional_1_layer_call_and_return_conditional_losses_57379410δl9:@ABCPQWXYZghnopq~¨©²³ΉΊ»ΌΕΖΟΠΦΧΨΩβγμνστυφ?D’A
:’7
-*

imageInput?????????`ΐ
p

 
ͺ ".’+
$!
0?????????`ΐ
 ³
J__inference_functional_1_layer_call_and_return_conditional_losses_57379584δl9:@ABCPQWXYZghnopq~¨©²³ΉΊ»ΌΕΖΟΠΦΧΨΩβγμνστυφ?D’A
:’7
-*

imageInput?????????`ΐ
p 

 
ͺ ".’+
$!
0?????????`ΐ
 ―
J__inference_functional_1_layer_call_and_return_conditional_losses_57380643ΰl9:@ABCPQWXYZghnopq~¨©²³ΉΊ»ΌΕΖΟΠΦΧΨΩβγμνστυφ?@’=
6’3
)&
inputs?????????`ΐ
p

 
ͺ ".’+
$!
0?????????`ΐ
 ―
J__inference_functional_1_layer_call_and_return_conditional_losses_57380934ΰl9:@ABCPQWXYZghnopq~¨©²³ΉΊ»ΌΕΖΟΠΦΧΨΩβγμνστυφ?@’=
6’3
)&
inputs?????????`ΐ
p 

 
ͺ ".’+
$!
0?????????`ΐ
 
/__inference_functional_1_layer_call_fn_57379892Χl9:@ABCPQWXYZghnopq~¨©²³ΉΊ»ΌΕΖΟΠΦΧΨΩβγμνστυφ?D’A
:’7
-*

imageInput?????????`ΐ
p

 
ͺ "!?????????`ΐ
/__inference_functional_1_layer_call_fn_57380199Χl9:@ABCPQWXYZghnopq~¨©²³ΉΊ»ΌΕΖΟΠΦΧΨΩβγμνστυφ?D’A
:’7
-*

imageInput?????????`ΐ
p 

 
ͺ "!?????????`ΐ
/__inference_functional_1_layer_call_fn_57381067Σl9:@ABCPQWXYZghnopq~¨©²³ΉΊ»ΌΕΖΟΠΦΧΨΩβγμνστυφ?@’=
6’3
)&
inputs?????????`ΐ
p

 
ͺ "!?????????`ΐ
/__inference_functional_1_layer_call_fn_57381200Σl9:@ABCPQWXYZghnopq~¨©²³ΉΊ»ΌΕΖΟΠΦΧΨΩβγμνστυφ?@’=
6’3
)&
inputs?????????`ΐ
p 

 
ͺ "!?????????`ΐπ
M__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_57377305R’O
H’E
C@
inputs4????????????????????????????????????
ͺ "H’E
>;
04????????????????????????????????????
 Θ
2__inference_max_pooling2d_1_layer_call_fn_57377311R’O
H’E
C@
inputs4????????????????????????????????????
ͺ ";84????????????????????????????????????π
M__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_57377421R’O
H’E
C@
inputs4????????????????????????????????????
ͺ "H’E
>;
04????????????????????????????????????
 Θ
2__inference_max_pooling2d_2_layer_call_fn_57377427R’O
H’E
C@
inputs4????????????????????????????????????
ͺ ";84????????????????????????????????????π
M__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_57377537R’O
H’E
C@
inputs4????????????????????????????????????
ͺ "H’E
>;
04????????????????????????????????????
 Θ
2__inference_max_pooling2d_3_layer_call_fn_57377543R’O
H’E
C@
inputs4????????????????????????????????????
ͺ ";84????????????????????????????????????ξ
K__inference_max_pooling2d_layer_call_and_return_conditional_losses_57377189R’O
H’E
C@
inputs4????????????????????????????????????
ͺ "H’E
>;
04????????????????????????????????????
 Ζ
0__inference_max_pooling2d_layer_call_fn_57377195R’O
H’E
C@
inputs4????????????????????????????????????
ͺ ";84????????????????????????????????????±
E__inference_re_lu_1_layer_call_and_return_conditional_losses_57381531h7’4
-’*
(%
inputs?????????0` 
ͺ "-’*
# 
0?????????0` 
 
*__inference_re_lu_1_layer_call_fn_57381536[7’4
-’*
(%
inputs?????????0` 
ͺ " ?????????0` ±
E__inference_re_lu_2_layer_call_and_return_conditional_losses_57381688h7’4
-’*
(%
inputs?????????0@
ͺ "-’*
# 
0?????????0@
 
*__inference_re_lu_2_layer_call_fn_57381693[7’4
-’*
(%
inputs?????????0@
ͺ " ?????????0@³
E__inference_re_lu_3_layer_call_and_return_conditional_losses_57381845j8’5
.’+
)&
inputs?????????
ͺ ".’+
$!
0?????????
 
*__inference_re_lu_3_layer_call_fn_57381850]8’5
.’+
)&
inputs?????????
ͺ "!?????????³
E__inference_re_lu_4_layer_call_and_return_conditional_losses_57382002j8’5
.’+
)&
inputs?????????
ͺ ".’+
$!
0?????????
 
*__inference_re_lu_4_layer_call_fn_57382007]8’5
.’+
)&
inputs?????????
ͺ "!?????????³
E__inference_re_lu_5_layer_call_and_return_conditional_losses_57382172j8’5
.’+
)&
inputs?????????
ͺ ".’+
$!
0?????????
 
*__inference_re_lu_5_layer_call_fn_57382177]8’5
.’+
)&
inputs?????????
ͺ "!?????????±
E__inference_re_lu_6_layer_call_and_return_conditional_losses_57382342h7’4
-’*
(%
inputs?????????0@
ͺ "-’*
# 
0?????????0@
 
*__inference_re_lu_6_layer_call_fn_57382347[7’4
-’*
(%
inputs?????????0@
ͺ " ?????????0@±
E__inference_re_lu_7_layer_call_and_return_conditional_losses_57382512h7’4
-’*
(%
inputs?????????0` 
ͺ "-’*
# 
0?????????0` 
 
*__inference_re_lu_7_layer_call_fn_57382517[7’4
-’*
(%
inputs?????????0` 
ͺ " ?????????0` ³
E__inference_re_lu_8_layer_call_and_return_conditional_losses_57382682j8’5
.’+
)&
inputs?????????`ΐ
ͺ ".’+
$!
0?????????`ΐ
 
*__inference_re_lu_8_layer_call_fn_57382687]8’5
.’+
)&
inputs?????????`ΐ
ͺ "!?????????`ΐ±
C__inference_re_lu_layer_call_and_return_conditional_losses_57381374j8’5
.’+
)&
inputs?????????`ΐ
ͺ ".’+
$!
0?????????`ΐ
 
(__inference_re_lu_layer_call_fn_57381379]8’5
.’+
)&
inputs?????????`ΐ
ͺ "!?????????`ΐ
&__inference_signature_wrapper_57380334ςl9:@ABCPQWXYZghnopq~¨©²³ΉΊ»ΌΕΖΟΠΦΧΨΩβγμνστυφ?J’G
’ 
@ͺ=
;

imageInput-*

imageInput?????????`ΐ"6ͺ3
1
final(%
final?????????`ΐΏ
Q__inference_tf_op_layer_RealDiv_layer_call_and_return_conditional_losses_57381206j8’5
.’+
)&
inputs?????????`ΐ
ͺ ".’+
$!
0?????????`ΐ
 
6__inference_tf_op_layer_RealDiv_layer_call_fn_57381211]8’5
.’+
)&
inputs?????????`ΐ
ͺ "!?????????`ΐ»
M__inference_tf_op_layer_Sub_layer_call_and_return_conditional_losses_57381217j8’5
.’+
)&
inputs?????????`ΐ
ͺ ".’+
$!
0?????????`ΐ
 
2__inference_tf_op_layer_Sub_layer_call_fn_57381222]8’5
.’+
)&
inputs?????????`ΐ
ͺ "!?????????`ΐ
R__inference_tf_op_layer_concat_6_layer_call_and_return_conditional_losses_57382014°~’{
t’q
ol
=:
inputs/0,???????????????????????????
+(
inputs/1?????????
ͺ ".’+
$!
0?????????
 ί
7__inference_tf_op_layer_concat_6_layer_call_fn_57382020£~’{
t’q
ol
=:
inputs/0,???????????????????????????
+(
inputs/1?????????
ͺ "!?????????
R__inference_tf_op_layer_concat_7_layer_call_and_return_conditional_losses_57382184?|’y
r’o
mj
<9
inputs/0+???????????????????????????@
*'
inputs/1?????????0@
ͺ ".’+
$!
0?????????0
 έ
7__inference_tf_op_layer_concat_7_layer_call_fn_57382190‘|’y
r’o
mj
<9
inputs/0+???????????????????????????@
*'
inputs/1?????????0@
ͺ "!?????????0
R__inference_tf_op_layer_concat_8_layer_call_and_return_conditional_losses_57382354­|’y
r’o
mj
<9
inputs/0+??????????????????????????? 
*'
inputs/1?????????0` 
ͺ "-’*
# 
0?????????0`@
 ά
7__inference_tf_op_layer_concat_8_layer_call_fn_57382360 |’y
r’o
mj
<9
inputs/0+??????????????????????????? 
*'
inputs/1?????????0` 
ͺ " ?????????0`@
R__inference_tf_op_layer_concat_9_layer_call_and_return_conditional_losses_57382524―}’z
s’p
nk
<9
inputs/0+???????????????????????????
+(
inputs/1?????????`ΐ
ͺ ".’+
$!
0?????????`ΐ 
 ή
7__inference_tf_op_layer_concat_9_layer_call_fn_57382530’}’z
s’p
nk
<9
inputs/0+???????????????????????????
+(
inputs/1?????????`ΐ
ͺ "!?????????`ΐ α
H__inference_upsample_6_layer_call_and_return_conditional_losses_57377685¨©J’G
@’=
;8
inputs,???????????????????????????
ͺ "@’=
63
0,???????????????????????????
 Ή
-__inference_upsample_6_layer_call_fn_57377695¨©J’G
@’=
;8
inputs,???????????????????????????
ͺ "30,???????????????????????????ΰ
H__inference_upsample_7_layer_call_and_return_conditional_losses_57377837ΕΖJ’G
@’=
;8
inputs,???????????????????????????
ͺ "?’<
52
0+???????????????????????????@
 Έ
-__inference_upsample_7_layer_call_fn_57377847ΕΖJ’G
@’=
;8
inputs,???????????????????????????
ͺ "2/+???????????????????????????@ί
H__inference_upsample_8_layer_call_and_return_conditional_losses_57377989βγI’F
?’<
:7
inputs+???????????????????????????@
ͺ "?’<
52
0+??????????????????????????? 
 ·
-__inference_upsample_8_layer_call_fn_57377999βγI’F
?’<
:7
inputs+???????????????????????????@
ͺ "2/+??????????????????????????? ί
H__inference_upsample_9_layer_call_and_return_conditional_losses_57378141?I’F
?’<
:7
inputs+??????????????????????????? 
ͺ "?’<
52
0+???????????????????????????
 ·
-__inference_upsample_9_layer_call_fn_57378151?I’F
?’<
:7
inputs+??????????????????????????? 
ͺ "2/+???????????????????????????