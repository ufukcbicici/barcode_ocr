ùª0
Ñ£
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
¾
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
 "serve*2.3.02v2.3.0-rc2-23-gb36436b0878Ï%
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
¢
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
¢
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
¢
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
¢
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
¢
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
¸
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*Ê·
value¿·B»· B³·
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
¡	variables
¢trainable_variables
£	keras_api
V
¤regularization_losses
¥	variables
¦trainable_variables
§	keras_api
n
¨kernel
	©bias
ªregularization_losses
«	variables
¬trainable_variables
­	keras_api
V
®regularization_losses
¯	variables
°trainable_variables
±	keras_api
n
²kernel
	³bias
´regularization_losses
µ	variables
¶trainable_variables
·	keras_api
 
	¸axis

¹gamma
	ºbeta
»moving_mean
¼moving_variance
½regularization_losses
¾	variables
¿trainable_variables
À	keras_api
V
Áregularization_losses
Â	variables
Ãtrainable_variables
Ä	keras_api
n
Åkernel
	Æbias
Çregularization_losses
È	variables
Étrainable_variables
Ê	keras_api
V
Ëregularization_losses
Ì	variables
Ítrainable_variables
Î	keras_api
n
Ïkernel
	Ðbias
Ñregularization_losses
Ò	variables
Ótrainable_variables
Ô	keras_api
 
	Õaxis

Ögamma
	×beta
Ømoving_mean
Ùmoving_variance
Úregularization_losses
Û	variables
Ütrainable_variables
Ý	keras_api
V
Þregularization_losses
ß	variables
àtrainable_variables
á	keras_api
n
âkernel
	ãbias
äregularization_losses
å	variables
ætrainable_variables
ç	keras_api
V
èregularization_losses
é	variables
êtrainable_variables
ë	keras_api
n
ìkernel
	íbias
îregularization_losses
ï	variables
ðtrainable_variables
ñ	keras_api
 
	òaxis

ógamma
	ôbeta
õmoving_mean
ömoving_variance
÷regularization_losses
ø	variables
ùtrainable_variables
ú	keras_api
V
ûregularization_losses
ü	variables
ýtrainable_variables
þ	keras_api
n
ÿkernel
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
¡	keras_api
 
¢
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
¹34
º35
»36
¼37
Å38
Æ39
Ï40
Ð41
Ö42
×43
Ø44
Ù45
â46
ã47
ì48
í49
ó50
ô51
õ52
ö53
ÿ54
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
¹24
º25
Å26
Æ27
Ï28
Ð29
Ö30
×31
â32
ã33
ì34
í35
ó36
ô37
ÿ38
39
40
41
42
43
44
45
²
,regularization_losses
¢non_trainable_variables
-	variables
£metrics
 ¤layer_regularization_losses
¥layer_metrics
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
ªlayer_metrics
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
 ®layer_regularization_losses
¯layer_metrics
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
´layer_metrics
µlayers
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
¶non_trainable_variables
E	variables
·metrics
 ¸layer_regularization_losses
¹layer_metrics
ºlayers
Ftrainable_variables
 
 
 
²
Hregularization_losses
»non_trainable_variables
I	variables
¼metrics
 ½layer_regularization_losses
¾layer_metrics
¿layers
Jtrainable_variables
 
 
 
²
Lregularization_losses
Ànon_trainable_variables
M	variables
Ámetrics
 Âlayer_regularization_losses
Ãlayer_metrics
Älayers
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
Ånon_trainable_variables
S	variables
Æmetrics
 Çlayer_regularization_losses
Èlayer_metrics
Élayers
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
Ênon_trainable_variables
\	variables
Ëmetrics
 Ìlayer_regularization_losses
Ílayer_metrics
Îlayers
]trainable_variables
 
 
 
²
_regularization_losses
Ïnon_trainable_variables
`	variables
Ðmetrics
 Ñlayer_regularization_losses
Òlayer_metrics
Ólayers
atrainable_variables
 
 
 
²
cregularization_losses
Ônon_trainable_variables
d	variables
Õmetrics
 Ölayer_regularization_losses
×layer_metrics
Ølayers
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
Ùnon_trainable_variables
j	variables
Úmetrics
 Ûlayer_regularization_losses
Ülayer_metrics
Ýlayers
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
Þnon_trainable_variables
s	variables
ßmetrics
 àlayer_regularization_losses
álayer_metrics
âlayers
ttrainable_variables
 
 
 
²
vregularization_losses
ãnon_trainable_variables
w	variables
ämetrics
 ålayer_regularization_losses
ælayer_metrics
çlayers
xtrainable_variables
 
 
 
²
zregularization_losses
ènon_trainable_variables
{	variables
émetrics
 êlayer_regularization_losses
ëlayer_metrics
ìlayers
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
µ
regularization_losses
ínon_trainable_variables
	variables
îmetrics
 ïlayer_regularization_losses
ðlayer_metrics
ñlayers
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
µ
regularization_losses
ònon_trainable_variables
	variables
ómetrics
 ôlayer_regularization_losses
õlayer_metrics
ölayers
trainable_variables
 
 
 
µ
regularization_losses
÷non_trainable_variables
	variables
ømetrics
 ùlayer_regularization_losses
úlayer_metrics
ûlayers
trainable_variables
 
 
 
µ
regularization_losses
ünon_trainable_variables
	variables
ýmetrics
 þlayer_regularization_losses
ÿlayer_metrics
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
µ
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
µ
 regularization_losses
non_trainable_variables
¡	variables
metrics
 layer_regularization_losses
layer_metrics
layers
¢trainable_variables
 
 
 
µ
¤regularization_losses
non_trainable_variables
¥	variables
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
µ
ªregularization_losses
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
µ
®regularization_losses
non_trainable_variables
¯	variables
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
µ
´regularization_losses
non_trainable_variables
µ	variables
metrics
 layer_regularization_losses
layer_metrics
layers
¶trainable_variables
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
 
¹0
º1
»2
¼3

¹0
º1
µ
½regularization_losses
non_trainable_variables
¾	variables
 metrics
 ¡layer_regularization_losses
¢layer_metrics
£layers
¿trainable_variables
 
 
 
µ
Áregularization_losses
¤non_trainable_variables
Â	variables
¥metrics
 ¦layer_regularization_losses
§layer_metrics
¨layers
Ãtrainable_variables
^\
VARIABLE_VALUEupsample_7/kernel7layer_with_weights-13/kernel/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEupsample_7/bias5layer_with_weights-13/bias/.ATTRIBUTES/VARIABLE_VALUE
 

Å0
Æ1

Å0
Æ1
µ
Çregularization_losses
©non_trainable_variables
È	variables
ªmetrics
 «layer_regularization_losses
¬layer_metrics
­layers
Étrainable_variables
 
 
 
µ
Ëregularization_losses
®non_trainable_variables
Ì	variables
¯metrics
 °layer_regularization_losses
±layer_metrics
²layers
Ítrainable_variables
ZX
VARIABLE_VALUEconv_7/kernel7layer_with_weights-14/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEconv_7/bias5layer_with_weights-14/bias/.ATTRIBUTES/VARIABLE_VALUE
 

Ï0
Ð1

Ï0
Ð1
µ
Ñregularization_losses
³non_trainable_variables
Ò	variables
´metrics
 µlayer_regularization_losses
¶layer_metrics
·layers
Ótrainable_variables
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
 
Ö0
×1
Ø2
Ù3

Ö0
×1
µ
Úregularization_losses
¸non_trainable_variables
Û	variables
¹metrics
 ºlayer_regularization_losses
»layer_metrics
¼layers
Ütrainable_variables
 
 
 
µ
Þregularization_losses
½non_trainable_variables
ß	variables
¾metrics
 ¿layer_regularization_losses
Àlayer_metrics
Álayers
àtrainable_variables
^\
VARIABLE_VALUEupsample_8/kernel7layer_with_weights-16/kernel/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEupsample_8/bias5layer_with_weights-16/bias/.ATTRIBUTES/VARIABLE_VALUE
 

â0
ã1

â0
ã1
µ
äregularization_losses
Ânon_trainable_variables
å	variables
Ãmetrics
 Älayer_regularization_losses
Ålayer_metrics
Ælayers
ætrainable_variables
 
 
 
µ
èregularization_losses
Çnon_trainable_variables
é	variables
Èmetrics
 Élayer_regularization_losses
Êlayer_metrics
Ëlayers
êtrainable_variables
ZX
VARIABLE_VALUEconv_8/kernel7layer_with_weights-17/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEconv_8/bias5layer_with_weights-17/bias/.ATTRIBUTES/VARIABLE_VALUE
 

ì0
í1

ì0
í1
µ
îregularization_losses
Ìnon_trainable_variables
ï	variables
Ímetrics
 Îlayer_regularization_losses
Ïlayer_metrics
Ðlayers
ðtrainable_variables
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
 
ó0
ô1
õ2
ö3

ó0
ô1
µ
÷regularization_losses
Ñnon_trainable_variables
ø	variables
Òmetrics
 Ólayer_regularization_losses
Ôlayer_metrics
Õlayers
ùtrainable_variables
 
 
 
µ
ûregularization_losses
Önon_trainable_variables
ü	variables
×metrics
 Ølayer_regularization_losses
Ùlayer_metrics
Úlayers
ýtrainable_variables
^\
VARIABLE_VALUEupsample_9/kernel7layer_with_weights-19/kernel/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEupsample_9/bias5layer_with_weights-19/bias/.ATTRIBUTES/VARIABLE_VALUE
 

ÿ0
1

ÿ0
1
µ
regularization_losses
Ûnon_trainable_variables
	variables
Ümetrics
 Ýlayer_regularization_losses
Þlayer_metrics
ßlayers
trainable_variables
 
 
 
µ
regularization_losses
ànon_trainable_variables
	variables
ámetrics
 âlayer_regularization_losses
ãlayer_metrics
älayers
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
µ
regularization_losses
ånon_trainable_variables
	variables
æmetrics
 çlayer_regularization_losses
èlayer_metrics
élayers
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
µ
regularization_losses
ênon_trainable_variables
	variables
ëmetrics
 ìlayer_regularization_losses
ílayer_metrics
îlayers
trainable_variables
 
 
 
µ
regularization_losses
ïnon_trainable_variables
	variables
ðmetrics
 ñlayer_regularization_losses
òlayer_metrics
ólayers
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
µ
regularization_losses
ônon_trainable_variables
	variables
õmetrics
 ölayer_regularization_losses
÷layer_metrics
ølayers
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
¼11
Ø12
Ù13
õ14
ö15
16
17
 
 
 
Î
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
¼1
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
Ø0
Ù1
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
õ0
ö1
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
:ÿÿÿÿÿÿÿÿÿ`À*
dtype0*%
shape:ÿÿÿÿÿÿÿÿÿ`À
ÿ
StatefulPartitionedCallStatefulPartitionedCallserving_default_imageInputconv_1/kernelconv_1/biasbatch_normalization/gammabatch_normalization/betabatch_normalization/moving_mean#batch_normalization/moving_varianceconv_2/kernelconv_2/biasbatch_normalization_1/gammabatch_normalization_1/beta!batch_normalization_1/moving_mean%batch_normalization_1/moving_varianceconv_3/kernelconv_3/biasbatch_normalization_2/gammabatch_normalization_2/beta!batch_normalization_2/moving_mean%batch_normalization_2/moving_varianceconv_4/kernelconv_4/biasbatch_normalization_3/gammabatch_normalization_3/beta!batch_normalization_3/moving_mean%batch_normalization_3/moving_varianceconv_5/kernelconv_5/biasbatch_normalization_4/gammabatch_normalization_4/beta!batch_normalization_4/moving_mean%batch_normalization_4/moving_varianceupsample_6/kernelupsample_6/biasconv_6/kernelconv_6/biasbatch_normalization_5/gammabatch_normalization_5/beta!batch_normalization_5/moving_mean%batch_normalization_5/moving_varianceupsample_7/kernelupsample_7/biasconv_7/kernelconv_7/biasbatch_normalization_6/gammabatch_normalization_6/beta!batch_normalization_6/moving_mean%batch_normalization_6/moving_varianceupsample_8/kernelupsample_8/biasconv_8/kernelconv_8/biasbatch_normalization_7/gammabatch_normalization_7/beta!batch_normalization_7/moving_mean%batch_normalization_7/moving_varianceupsample_9/kernelupsample_9/biasconv_9/kernelconv_9/biasbatch_normalization_8/gammabatch_normalization_8/beta!batch_normalization_8/moving_mean%batch_normalization_8/moving_variancefinal/kernel
final/bias*L
TinE
C2A*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`À*b
_read_only_resource_inputsD
B@	
 !"#$%&'()*+,-./0123456789:;<=>?@*2
config_proto" 

CPU

GPU2 *0J 8 *0
f+R)
'__inference_signature_wrapper_107947054
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 

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
GPU2 *0J 8 *+
f&R$
"__inference__traced_save_107949641

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
GPU2 *0J 8 *.
f)R'
%__inference__traced_restore_107949843ÑÔ"
¡~
è
"__inference__traced_save_107949641
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

identity_1¢MergeV2Checkpoints
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
value3B1 B+_temp_97cc2c85a6bd497fbba905f46c54feb3/part2	
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
SaveV2/shape_and_slicesó
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0(savev2_conv_1_kernel_read_readvariableop&savev2_conv_1_bias_read_readvariableop4savev2_batch_normalization_gamma_read_readvariableop3savev2_batch_normalization_beta_read_readvariableop:savev2_batch_normalization_moving_mean_read_readvariableop>savev2_batch_normalization_moving_variance_read_readvariableop(savev2_conv_2_kernel_read_readvariableop&savev2_conv_2_bias_read_readvariableop6savev2_batch_normalization_1_gamma_read_readvariableop5savev2_batch_normalization_1_beta_read_readvariableop<savev2_batch_normalization_1_moving_mean_read_readvariableop@savev2_batch_normalization_1_moving_variance_read_readvariableop(savev2_conv_3_kernel_read_readvariableop&savev2_conv_3_bias_read_readvariableop6savev2_batch_normalization_2_gamma_read_readvariableop5savev2_batch_normalization_2_beta_read_readvariableop<savev2_batch_normalization_2_moving_mean_read_readvariableop@savev2_batch_normalization_2_moving_variance_read_readvariableop(savev2_conv_4_kernel_read_readvariableop&savev2_conv_4_bias_read_readvariableop6savev2_batch_normalization_3_gamma_read_readvariableop5savev2_batch_normalization_3_beta_read_readvariableop<savev2_batch_normalization_3_moving_mean_read_readvariableop@savev2_batch_normalization_3_moving_variance_read_readvariableop(savev2_conv_5_kernel_read_readvariableop&savev2_conv_5_bias_read_readvariableop6savev2_batch_normalization_4_gamma_read_readvariableop5savev2_batch_normalization_4_beta_read_readvariableop<savev2_batch_normalization_4_moving_mean_read_readvariableop@savev2_batch_normalization_4_moving_variance_read_readvariableop,savev2_upsample_6_kernel_read_readvariableop*savev2_upsample_6_bias_read_readvariableop(savev2_conv_6_kernel_read_readvariableop&savev2_conv_6_bias_read_readvariableop6savev2_batch_normalization_5_gamma_read_readvariableop5savev2_batch_normalization_5_beta_read_readvariableop<savev2_batch_normalization_5_moving_mean_read_readvariableop@savev2_batch_normalization_5_moving_variance_read_readvariableop,savev2_upsample_7_kernel_read_readvariableop*savev2_upsample_7_bias_read_readvariableop(savev2_conv_7_kernel_read_readvariableop&savev2_conv_7_bias_read_readvariableop6savev2_batch_normalization_6_gamma_read_readvariableop5savev2_batch_normalization_6_beta_read_readvariableop<savev2_batch_normalization_6_moving_mean_read_readvariableop@savev2_batch_normalization_6_moving_variance_read_readvariableop,savev2_upsample_8_kernel_read_readvariableop*savev2_upsample_8_bias_read_readvariableop(savev2_conv_8_kernel_read_readvariableop&savev2_conv_8_bias_read_readvariableop6savev2_batch_normalization_7_gamma_read_readvariableop5savev2_batch_normalization_7_beta_read_readvariableop<savev2_batch_normalization_7_moving_mean_read_readvariableop@savev2_batch_normalization_7_moving_variance_read_readvariableop,savev2_upsample_9_kernel_read_readvariableop*savev2_upsample_9_bias_read_readvariableop(savev2_conv_9_kernel_read_readvariableop&savev2_conv_9_bias_read_readvariableop6savev2_batch_normalization_8_gamma_read_readvariableop5savev2_batch_normalization_8_beta_read_readvariableop<savev2_batch_normalization_8_moving_mean_read_readvariableop@savev2_batch_normalization_8_moving_variance_read_readvariableop'savev2_final_kernel_read_readvariableop%savev2_final_bias_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *O
dtypesE
C2A2
SaveV2º
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes¡
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

identity_1Identity_1:output:0*Ú
_input_shapesÈ
Å: ::::::: : : : : : : @:@:@:@:@:@:@::::::::::::::::::::@:@:@:@:@:@:@:@: @: :@ : : : : : : :: :::::::: 2(
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
Ø
±
T__inference_batch_normalization_5_layer_call_and_return_conditional_losses_107944477

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity¢AssignNewValue¢AssignNewValue_1u
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
FusedBatchNormV3/ReadVariableOp®
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ï
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3ÿ
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
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

d
8__inference_tf_op_layer_concat_7_layer_call_fn_107948910
inputs_0
inputs_1
identityì
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *\
fWRU
S__inference_tf_op_layer_concat_7_layer_call_and_return_conditional_losses_1079457162
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ02

Identity"
identityIdentity:output:0*[
_input_shapesJ
H:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ0@:k g
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@
"
_user_specified_name
inputs/1

±
T__inference_batch_normalization_7_layer_call_and_return_conditional_losses_107949183

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity¢AssignNewValue¢AssignNewValue_1t
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
!FusedBatchNormV3/ReadVariableOp_1Ø
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ0` : : : : :*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3ÿ
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
:ÿÿÿÿÿÿÿÿÿ0` 2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ0` ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0` 
 
_user_specified_nameinputs
Á
Ü
0__inference_functional_1_layer_call_fn_107947920

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
identity¢StatefulPartitionedCallä	
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
:ÿÿÿÿÿÿÿÿÿ`À*b
_read_only_resource_inputsD
B@	
 !"#$%&'()*+,-./0123456789:;<=>?@*2
config_proto" 

CPU

GPU2 *0J 8 *T
fORM
K__inference_functional_1_layer_call_and_return_conditional_losses_1079467882
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`À2

Identity"
identityIdentity:output:0*±
_input_shapes
:ÿÿÿÿÿÿÿÿÿ`À::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`À
 
_user_specified_nameinputs
õµ
ì
K__inference_functional_1_layer_call_and_return_conditional_losses_107947654

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
 *  ÿB2
tf_op_layer_RealDiv/RealDiv/y¿
tf_op_layer_RealDiv/RealDivRealDivinputs&tf_op_layer_RealDiv/RealDiv/y:output:0*
T0*
_cloned(*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`À2
tf_op_layer_RealDiv/RealDivs
tf_op_layer_Sub/Sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
tf_op_layer_Sub/Sub/y¼
tf_op_layer_Sub/SubSubtf_op_layer_RealDiv/RealDiv:z:0tf_op_layer_Sub/Sub/y:output:0*
T0*
_cloned(*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`À2
tf_op_layer_Sub/Subª
conv_1/Conv2D/ReadVariableOpReadVariableOp%conv_1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
conv_1/Conv2D/ReadVariableOpÊ
conv_1/Conv2DConv2Dtf_op_layer_Sub/Sub:z:0$conv_1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`À*
paddingSAME*
strides
2
conv_1/Conv2D¡
conv_1/BiasAdd/ReadVariableOpReadVariableOp&conv_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
conv_1/BiasAdd/ReadVariableOp¥
conv_1/BiasAddBiasAddconv_1/Conv2D:output:0%conv_1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`À2
conv_1/BiasAdd°
"batch_normalization/ReadVariableOpReadVariableOp+batch_normalization_readvariableop_resource*
_output_shapes
:*
dtype02$
"batch_normalization/ReadVariableOp¶
$batch_normalization/ReadVariableOp_1ReadVariableOp-batch_normalization_readvariableop_1_resource*
_output_shapes
:*
dtype02&
$batch_normalization/ReadVariableOp_1ã
3batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOp<batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype025
3batch_normalization/FusedBatchNormV3/ReadVariableOpé
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp>batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype027
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1Ô
$batch_normalization/FusedBatchNormV3FusedBatchNormV3conv_1/BiasAdd:output:0*batch_normalization/ReadVariableOp:value:0,batch_normalization/ReadVariableOp_1:value:0;batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0=batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*L
_output_shapes:
8:ÿÿÿÿÿÿÿÿÿ`À:::::*
epsilon%o:*
is_training( 2&
$batch_normalization/FusedBatchNormV3

re_lu/ReluRelu(batch_normalization/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`À2

re_lu/ReluÀ
max_pooling2d/MaxPoolMaxPoolre_lu/Relu:activations:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0`*
ksize
*
paddingVALID*
strides
2
max_pooling2d/MaxPoolª
conv_2/Conv2D/ReadVariableOpReadVariableOp%conv_2_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
conv_2/Conv2D/ReadVariableOpÐ
conv_2/Conv2DConv2Dmax_pooling2d/MaxPool:output:0$conv_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0` *
paddingSAME*
strides
2
conv_2/Conv2D¡
conv_2/BiasAdd/ReadVariableOpReadVariableOp&conv_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02
conv_2/BiasAdd/ReadVariableOp¤
conv_2/BiasAddBiasAddconv_2/Conv2D:output:0%conv_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0` 2
conv_2/BiasAdd¶
$batch_normalization_1/ReadVariableOpReadVariableOp-batch_normalization_1_readvariableop_resource*
_output_shapes
: *
dtype02&
$batch_normalization_1/ReadVariableOp¼
&batch_normalization_1/ReadVariableOp_1ReadVariableOp/batch_normalization_1_readvariableop_1_resource*
_output_shapes
: *
dtype02(
&batch_normalization_1/ReadVariableOp_1é
5batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype027
5batch_normalization_1/FusedBatchNormV3/ReadVariableOpï
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype029
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ß
&batch_normalization_1/FusedBatchNormV3FusedBatchNormV3conv_2/BiasAdd:output:0,batch_normalization_1/ReadVariableOp:value:0.batch_normalization_1/ReadVariableOp_1:value:0=batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ0` : : : : :*
epsilon%o:*
is_training( 2(
&batch_normalization_1/FusedBatchNormV3
re_lu_1/ReluRelu*batch_normalization_1/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0` 2
re_lu_1/ReluÆ
max_pooling2d_1/MaxPoolMaxPoolre_lu_1/Relu:activations:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0 *
ksize
*
paddingVALID*
strides
2
max_pooling2d_1/MaxPoolª
conv_3/Conv2D/ReadVariableOpReadVariableOp%conv_3_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02
conv_3/Conv2D/ReadVariableOpÒ
conv_3/Conv2DConv2D max_pooling2d_1/MaxPool:output:0$conv_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@*
paddingSAME*
strides
2
conv_3/Conv2D¡
conv_3/BiasAdd/ReadVariableOpReadVariableOp&conv_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
conv_3/BiasAdd/ReadVariableOp¤
conv_3/BiasAddBiasAddconv_3/Conv2D:output:0%conv_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@2
conv_3/BiasAdd¶
$batch_normalization_2/ReadVariableOpReadVariableOp-batch_normalization_2_readvariableop_resource*
_output_shapes
:@*
dtype02&
$batch_normalization_2/ReadVariableOp¼
&batch_normalization_2/ReadVariableOp_1ReadVariableOp/batch_normalization_2_readvariableop_1_resource*
_output_shapes
:@*
dtype02(
&batch_normalization_2/ReadVariableOp_1é
5batch_normalization_2/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype027
5batch_normalization_2/FusedBatchNormV3/ReadVariableOpï
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype029
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1ß
&batch_normalization_2/FusedBatchNormV3FusedBatchNormV3conv_3/BiasAdd:output:0,batch_normalization_2/ReadVariableOp:value:0.batch_normalization_2/ReadVariableOp_1:value:0=batch_normalization_2/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ0@:@:@:@:@:*
epsilon%o:*
is_training( 2(
&batch_normalization_2/FusedBatchNormV3
re_lu_2/ReluRelu*batch_normalization_2/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@2
re_lu_2/ReluÆ
max_pooling2d_2/MaxPoolMaxPoolre_lu_2/Relu:activations:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
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
conv_4/Conv2D/ReadVariableOpÓ
conv_4/Conv2DConv2D max_pooling2d_2/MaxPool:output:0$conv_4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
2
conv_4/Conv2D¢
conv_4/BiasAdd/ReadVariableOpReadVariableOp&conv_4_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
conv_4/BiasAdd/ReadVariableOp¥
conv_4/BiasAddBiasAddconv_4/Conv2D:output:0%conv_4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
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
&batch_normalization_3/ReadVariableOp_1ê
5batch_normalization_3/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_3_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype027
5batch_normalization_3/FusedBatchNormV3/ReadVariableOpð
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype029
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1ä
&batch_normalization_3/FusedBatchNormV3FusedBatchNormV3conv_4/BiasAdd:output:0,batch_normalization_3/ReadVariableOp:value:0.batch_normalization_3/ReadVariableOp_1:value:0=batch_normalization_3/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
is_training( 2(
&batch_normalization_3/FusedBatchNormV3
re_lu_3/ReluRelu*batch_normalization_3/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
re_lu_3/ReluÇ
max_pooling2d_3/MaxPoolMaxPoolre_lu_3/Relu:activations:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
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
conv_5/Conv2D/ReadVariableOpÓ
conv_5/Conv2DConv2D max_pooling2d_3/MaxPool:output:0$conv_5/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
2
conv_5/Conv2D¢
conv_5/BiasAdd/ReadVariableOpReadVariableOp&conv_5_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
conv_5/BiasAdd/ReadVariableOp¥
conv_5/BiasAddBiasAddconv_5/Conv2D:output:0%conv_5/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
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
&batch_normalization_4/ReadVariableOp_1ê
5batch_normalization_4/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_4_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype027
5batch_normalization_4/FusedBatchNormV3/ReadVariableOpð
7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype029
7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1ä
&batch_normalization_4/FusedBatchNormV3FusedBatchNormV3conv_5/BiasAdd:output:0,batch_normalization_4/ReadVariableOp:value:0.batch_normalization_4/ReadVariableOp_1:value:0=batch_normalization_4/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
is_training( 2(
&batch_normalization_4/FusedBatchNormV3
re_lu_4/ReluRelu*batch_normalization_4/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
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
 upsample_6/strided_slice/stack_2¤
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
upsample_6/stack/3Ô
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
"upsample_6/strided_slice_1/stack_2®
upsample_6/strided_slice_1StridedSliceupsample_6/stack:output:0)upsample_6/strided_slice_1/stack:output:0+upsample_6/strided_slice_1/stack_1:output:0+upsample_6/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
upsample_6/strided_slice_1Ö
*upsample_6/conv2d_transpose/ReadVariableOpReadVariableOp3upsample_6_conv2d_transpose_readvariableop_resource*(
_output_shapes
:*
dtype02,
*upsample_6/conv2d_transpose/ReadVariableOp 
upsample_6/conv2d_transposeConv2DBackpropInputupsample_6/stack:output:02upsample_6/conv2d_transpose/ReadVariableOp:value:0re_lu_4/Relu:activations:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides
2
upsample_6/conv2d_transpose®
!upsample_6/BiasAdd/ReadVariableOpReadVariableOp*upsample_6_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02#
!upsample_6/BiasAdd/ReadVariableOp¿
upsample_6/BiasAddBiasAdd$upsample_6/conv2d_transpose:output:0)upsample_6/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
upsample_6/BiasAdd
"tf_op_layer_concat_6/concat_6/axisConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2$
"tf_op_layer_concat_6/concat_6/axis
tf_op_layer_concat_6/concat_6ConcatV2upsample_6/BiasAdd:output:0re_lu_3/Relu:activations:0+tf_op_layer_concat_6/concat_6/axis:output:0*
N*
T0*
_cloned(*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tf_op_layer_concat_6/concat_6¬
conv_6/Conv2D/ReadVariableOpReadVariableOp%conv_6_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02
conv_6/Conv2D/ReadVariableOpÙ
conv_6/Conv2DConv2D&tf_op_layer_concat_6/concat_6:output:0$conv_6/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
2
conv_6/Conv2D¢
conv_6/BiasAdd/ReadVariableOpReadVariableOp&conv_6_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
conv_6/BiasAdd/ReadVariableOp¥
conv_6/BiasAddBiasAddconv_6/Conv2D:output:0%conv_6/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
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
&batch_normalization_5/ReadVariableOp_1ê
5batch_normalization_5/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_5_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype027
5batch_normalization_5/FusedBatchNormV3/ReadVariableOpð
7batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype029
7batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1ä
&batch_normalization_5/FusedBatchNormV3FusedBatchNormV3conv_6/BiasAdd:output:0,batch_normalization_5/ReadVariableOp:value:0.batch_normalization_5/ReadVariableOp_1:value:0=batch_normalization_5/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
is_training( 2(
&batch_normalization_5/FusedBatchNormV3
re_lu_5/ReluRelu*batch_normalization_5/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
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
 upsample_7/strided_slice/stack_2¤
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
upsample_7/stack/3Ô
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
"upsample_7/strided_slice_1/stack_2®
upsample_7/strided_slice_1StridedSliceupsample_7/stack:output:0)upsample_7/strided_slice_1/stack:output:0+upsample_7/strided_slice_1/stack_1:output:0+upsample_7/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
upsample_7/strided_slice_1Õ
*upsample_7/conv2d_transpose/ReadVariableOpReadVariableOp3upsample_7_conv2d_transpose_readvariableop_resource*'
_output_shapes
:@*
dtype02,
*upsample_7/conv2d_transpose/ReadVariableOp
upsample_7/conv2d_transposeConv2DBackpropInputupsample_7/stack:output:02upsample_7/conv2d_transpose/ReadVariableOp:value:0re_lu_5/Relu:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@*
paddingVALID*
strides
2
upsample_7/conv2d_transpose­
!upsample_7/BiasAdd/ReadVariableOpReadVariableOp*upsample_7_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02#
!upsample_7/BiasAdd/ReadVariableOp¾
upsample_7/BiasAddBiasAdd$upsample_7/conv2d_transpose:output:0)upsample_7/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@2
upsample_7/BiasAdd
"tf_op_layer_concat_7/concat_7/axisConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2$
"tf_op_layer_concat_7/concat_7/axis
tf_op_layer_concat_7/concat_7ConcatV2upsample_7/BiasAdd:output:0re_lu_2/Relu:activations:0+tf_op_layer_concat_7/concat_7/axis:output:0*
N*
T0*
_cloned(*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ02
tf_op_layer_concat_7/concat_7«
conv_7/Conv2D/ReadVariableOpReadVariableOp%conv_7_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype02
conv_7/Conv2D/ReadVariableOpØ
conv_7/Conv2DConv2D&tf_op_layer_concat_7/concat_7:output:0$conv_7/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@*
paddingSAME*
strides
2
conv_7/Conv2D¡
conv_7/BiasAdd/ReadVariableOpReadVariableOp&conv_7_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
conv_7/BiasAdd/ReadVariableOp¤
conv_7/BiasAddBiasAddconv_7/Conv2D:output:0%conv_7/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@2
conv_7/BiasAdd¶
$batch_normalization_6/ReadVariableOpReadVariableOp-batch_normalization_6_readvariableop_resource*
_output_shapes
:@*
dtype02&
$batch_normalization_6/ReadVariableOp¼
&batch_normalization_6/ReadVariableOp_1ReadVariableOp/batch_normalization_6_readvariableop_1_resource*
_output_shapes
:@*
dtype02(
&batch_normalization_6/ReadVariableOp_1é
5batch_normalization_6/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_6_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype027
5batch_normalization_6/FusedBatchNormV3/ReadVariableOpï
7batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype029
7batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1ß
&batch_normalization_6/FusedBatchNormV3FusedBatchNormV3conv_7/BiasAdd:output:0,batch_normalization_6/ReadVariableOp:value:0.batch_normalization_6/ReadVariableOp_1:value:0=batch_normalization_6/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ0@:@:@:@:@:*
epsilon%o:*
is_training( 2(
&batch_normalization_6/FusedBatchNormV3
re_lu_6/ReluRelu*batch_normalization_6/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@2
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
 upsample_8/strided_slice/stack_2¤
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
upsample_8/stack/3Ô
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
"upsample_8/strided_slice_1/stack_2®
upsample_8/strided_slice_1StridedSliceupsample_8/stack:output:0)upsample_8/strided_slice_1/stack:output:0+upsample_8/strided_slice_1/stack_1:output:0+upsample_8/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
upsample_8/strided_slice_1Ô
*upsample_8/conv2d_transpose/ReadVariableOpReadVariableOp3upsample_8_conv2d_transpose_readvariableop_resource*&
_output_shapes
: @*
dtype02,
*upsample_8/conv2d_transpose/ReadVariableOp
upsample_8/conv2d_transposeConv2DBackpropInputupsample_8/stack:output:02upsample_8/conv2d_transpose/ReadVariableOp:value:0re_lu_6/Relu:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0` *
paddingVALID*
strides
2
upsample_8/conv2d_transpose­
!upsample_8/BiasAdd/ReadVariableOpReadVariableOp*upsample_8_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02#
!upsample_8/BiasAdd/ReadVariableOp¾
upsample_8/BiasAddBiasAdd$upsample_8/conv2d_transpose:output:0)upsample_8/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0` 2
upsample_8/BiasAdd
"tf_op_layer_concat_8/concat_8/axisConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2$
"tf_op_layer_concat_8/concat_8/axis
tf_op_layer_concat_8/concat_8ConcatV2upsample_8/BiasAdd:output:0re_lu_1/Relu:activations:0+tf_op_layer_concat_8/concat_8/axis:output:0*
N*
T0*
_cloned(*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0`@2
tf_op_layer_concat_8/concat_8ª
conv_8/Conv2D/ReadVariableOpReadVariableOp%conv_8_conv2d_readvariableop_resource*&
_output_shapes
:@ *
dtype02
conv_8/Conv2D/ReadVariableOpØ
conv_8/Conv2DConv2D&tf_op_layer_concat_8/concat_8:output:0$conv_8/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0` *
paddingSAME*
strides
2
conv_8/Conv2D¡
conv_8/BiasAdd/ReadVariableOpReadVariableOp&conv_8_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02
conv_8/BiasAdd/ReadVariableOp¤
conv_8/BiasAddBiasAddconv_8/Conv2D:output:0%conv_8/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0` 2
conv_8/BiasAdd¶
$batch_normalization_7/ReadVariableOpReadVariableOp-batch_normalization_7_readvariableop_resource*
_output_shapes
: *
dtype02&
$batch_normalization_7/ReadVariableOp¼
&batch_normalization_7/ReadVariableOp_1ReadVariableOp/batch_normalization_7_readvariableop_1_resource*
_output_shapes
: *
dtype02(
&batch_normalization_7/ReadVariableOp_1é
5batch_normalization_7/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_7_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype027
5batch_normalization_7/FusedBatchNormV3/ReadVariableOpï
7batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype029
7batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1ß
&batch_normalization_7/FusedBatchNormV3FusedBatchNormV3conv_8/BiasAdd:output:0,batch_normalization_7/ReadVariableOp:value:0.batch_normalization_7/ReadVariableOp_1:value:0=batch_normalization_7/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ0` : : : : :*
epsilon%o:*
is_training( 2(
&batch_normalization_7/FusedBatchNormV3
re_lu_7/ReluRelu*batch_normalization_7/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0` 2
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
 upsample_9/strided_slice/stack_2¤
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
B :À2
upsample_9/stack/2j
upsample_9/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2
upsample_9/stack/3Ô
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
"upsample_9/strided_slice_1/stack_2®
upsample_9/strided_slice_1StridedSliceupsample_9/stack:output:0)upsample_9/strided_slice_1/stack:output:0+upsample_9/strided_slice_1/stack_1:output:0+upsample_9/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
upsample_9/strided_slice_1Ô
*upsample_9/conv2d_transpose/ReadVariableOpReadVariableOp3upsample_9_conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype02,
*upsample_9/conv2d_transpose/ReadVariableOp 
upsample_9/conv2d_transposeConv2DBackpropInputupsample_9/stack:output:02upsample_9/conv2d_transpose/ReadVariableOp:value:0re_lu_7/Relu:activations:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`À*
paddingVALID*
strides
2
upsample_9/conv2d_transpose­
!upsample_9/BiasAdd/ReadVariableOpReadVariableOp*upsample_9_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02#
!upsample_9/BiasAdd/ReadVariableOp¿
upsample_9/BiasAddBiasAdd$upsample_9/conv2d_transpose:output:0)upsample_9/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`À2
upsample_9/BiasAdd
"tf_op_layer_concat_9/concat_9/axisConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2$
"tf_op_layer_concat_9/concat_9/axis
tf_op_layer_concat_9/concat_9ConcatV2upsample_9/BiasAdd:output:0re_lu/Relu:activations:0+tf_op_layer_concat_9/concat_9/axis:output:0*
N*
T0*
_cloned(*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`À 2
tf_op_layer_concat_9/concat_9ª
conv_9/Conv2D/ReadVariableOpReadVariableOp%conv_9_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
conv_9/Conv2D/ReadVariableOpÙ
conv_9/Conv2DConv2D&tf_op_layer_concat_9/concat_9:output:0$conv_9/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`À*
paddingSAME*
strides
2
conv_9/Conv2D¡
conv_9/BiasAdd/ReadVariableOpReadVariableOp&conv_9_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
conv_9/BiasAdd/ReadVariableOp¥
conv_9/BiasAddBiasAddconv_9/Conv2D:output:0%conv_9/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`À2
conv_9/BiasAdd¶
$batch_normalization_8/ReadVariableOpReadVariableOp-batch_normalization_8_readvariableop_resource*
_output_shapes
:*
dtype02&
$batch_normalization_8/ReadVariableOp¼
&batch_normalization_8/ReadVariableOp_1ReadVariableOp/batch_normalization_8_readvariableop_1_resource*
_output_shapes
:*
dtype02(
&batch_normalization_8/ReadVariableOp_1é
5batch_normalization_8/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_8_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype027
5batch_normalization_8/FusedBatchNormV3/ReadVariableOpï
7batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype029
7batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1à
&batch_normalization_8/FusedBatchNormV3FusedBatchNormV3conv_9/BiasAdd:output:0,batch_normalization_8/ReadVariableOp:value:0.batch_normalization_8/ReadVariableOp_1:value:0=batch_normalization_8/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*L
_output_shapes:
8:ÿÿÿÿÿÿÿÿÿ`À:::::*
epsilon%o:*
is_training( 2(
&batch_normalization_8/FusedBatchNormV3
re_lu_8/ReluRelu*batch_normalization_8/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`À2
re_lu_8/Relu§
final/Conv2D/ReadVariableOpReadVariableOp$final_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
final/Conv2D/ReadVariableOpÊ
final/Conv2DConv2Dre_lu_8/Relu:activations:0#final/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`À*
paddingSAME*
strides
2
final/Conv2D
final/BiasAdd/ReadVariableOpReadVariableOp%final_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
final/BiasAdd/ReadVariableOp¡
final/BiasAddBiasAddfinal/Conv2D:output:0$final/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`À2
final/BiasAdds
IdentityIdentityfinal/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`À2

Identity"
identityIdentity:output:0*±
_input_shapes
:ÿÿÿÿÿÿÿÿÿ`À:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`À
 
_user_specified_nameinputs
ß

T__inference_batch_normalization_3_layer_call_and_return_conditional_losses_107948534

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
FusedBatchNormV3/ReadVariableOp®
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ï
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
is_training( 2
FusedBatchNormV3q
IdentityIdentityFusedBatchNormV3:y:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿ:::::X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ì
±
T__inference_batch_normalization_2_layer_call_and_return_conditional_losses_107944093

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity¢AssignNewValue¢AssignNewValue_1t
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
!FusedBatchNormV3/ReadVariableOp_1ê
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@:@:@:@:@:*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3ÿ
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
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
òù
"
$__inference__wrapped_model_107943799

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
 *  ÿB2,
*functional_1/tf_op_layer_RealDiv/RealDiv/yê
(functional_1/tf_op_layer_RealDiv/RealDivRealDiv
imageinput3functional_1/tf_op_layer_RealDiv/RealDiv/y:output:0*
T0*
_cloned(*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`À2*
(functional_1/tf_op_layer_RealDiv/RealDiv
"functional_1/tf_op_layer_Sub/Sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2$
"functional_1/tf_op_layer_Sub/Sub/yð
 functional_1/tf_op_layer_Sub/SubSub,functional_1/tf_op_layer_RealDiv/RealDiv:z:0+functional_1/tf_op_layer_Sub/Sub/y:output:0*
T0*
_cloned(*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`À2"
 functional_1/tf_op_layer_Sub/SubÑ
)functional_1/conv_1/Conv2D/ReadVariableOpReadVariableOp2functional_1_conv_1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02+
)functional_1/conv_1/Conv2D/ReadVariableOpþ
functional_1/conv_1/Conv2DConv2D$functional_1/tf_op_layer_Sub/Sub:z:01functional_1/conv_1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`À*
paddingSAME*
strides
2
functional_1/conv_1/Conv2DÈ
*functional_1/conv_1/BiasAdd/ReadVariableOpReadVariableOp3functional_1_conv_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02,
*functional_1/conv_1/BiasAdd/ReadVariableOpÙ
functional_1/conv_1/BiasAddBiasAdd#functional_1/conv_1/Conv2D:output:02functional_1/conv_1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`À2
functional_1/conv_1/BiasAdd×
/functional_1/batch_normalization/ReadVariableOpReadVariableOp8functional_1_batch_normalization_readvariableop_resource*
_output_shapes
:*
dtype021
/functional_1/batch_normalization/ReadVariableOpÝ
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
Bfunctional_1/batch_normalization/FusedBatchNormV3/ReadVariableOp_1¯
1functional_1/batch_normalization/FusedBatchNormV3FusedBatchNormV3$functional_1/conv_1/BiasAdd:output:07functional_1/batch_normalization/ReadVariableOp:value:09functional_1/batch_normalization/ReadVariableOp_1:value:0Hfunctional_1/batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0Jfunctional_1/batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*L
_output_shapes:
8:ÿÿÿÿÿÿÿÿÿ`À:::::*
epsilon%o:*
is_training( 23
1functional_1/batch_normalization/FusedBatchNormV3¬
functional_1/re_lu/ReluRelu5functional_1/batch_normalization/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`À2
functional_1/re_lu/Reluç
"functional_1/max_pooling2d/MaxPoolMaxPool%functional_1/re_lu/Relu:activations:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0`*
ksize
*
paddingVALID*
strides
2$
"functional_1/max_pooling2d/MaxPoolÑ
)functional_1/conv_2/Conv2D/ReadVariableOpReadVariableOp2functional_1_conv_2_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02+
)functional_1/conv_2/Conv2D/ReadVariableOp
functional_1/conv_2/Conv2DConv2D+functional_1/max_pooling2d/MaxPool:output:01functional_1/conv_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0` *
paddingSAME*
strides
2
functional_1/conv_2/Conv2DÈ
*functional_1/conv_2/BiasAdd/ReadVariableOpReadVariableOp3functional_1_conv_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02,
*functional_1/conv_2/BiasAdd/ReadVariableOpØ
functional_1/conv_2/BiasAddBiasAdd#functional_1/conv_2/Conv2D:output:02functional_1/conv_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0` 2
functional_1/conv_2/BiasAddÝ
1functional_1/batch_normalization_1/ReadVariableOpReadVariableOp:functional_1_batch_normalization_1_readvariableop_resource*
_output_shapes
: *
dtype023
1functional_1/batch_normalization_1/ReadVariableOpã
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
Dfunctional_1/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1º
3functional_1/batch_normalization_1/FusedBatchNormV3FusedBatchNormV3$functional_1/conv_2/BiasAdd:output:09functional_1/batch_normalization_1/ReadVariableOp:value:0;functional_1/batch_normalization_1/ReadVariableOp_1:value:0Jfunctional_1/batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0Lfunctional_1/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ0` : : : : :*
epsilon%o:*
is_training( 25
3functional_1/batch_normalization_1/FusedBatchNormV3±
functional_1/re_lu_1/ReluRelu7functional_1/batch_normalization_1/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0` 2
functional_1/re_lu_1/Reluí
$functional_1/max_pooling2d_1/MaxPoolMaxPool'functional_1/re_lu_1/Relu:activations:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0 *
ksize
*
paddingVALID*
strides
2&
$functional_1/max_pooling2d_1/MaxPoolÑ
)functional_1/conv_3/Conv2D/ReadVariableOpReadVariableOp2functional_1_conv_3_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02+
)functional_1/conv_3/Conv2D/ReadVariableOp
functional_1/conv_3/Conv2DConv2D-functional_1/max_pooling2d_1/MaxPool:output:01functional_1/conv_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@*
paddingSAME*
strides
2
functional_1/conv_3/Conv2DÈ
*functional_1/conv_3/BiasAdd/ReadVariableOpReadVariableOp3functional_1_conv_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02,
*functional_1/conv_3/BiasAdd/ReadVariableOpØ
functional_1/conv_3/BiasAddBiasAdd#functional_1/conv_3/Conv2D:output:02functional_1/conv_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@2
functional_1/conv_3/BiasAddÝ
1functional_1/batch_normalization_2/ReadVariableOpReadVariableOp:functional_1_batch_normalization_2_readvariableop_resource*
_output_shapes
:@*
dtype023
1functional_1/batch_normalization_2/ReadVariableOpã
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
Dfunctional_1/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1º
3functional_1/batch_normalization_2/FusedBatchNormV3FusedBatchNormV3$functional_1/conv_3/BiasAdd:output:09functional_1/batch_normalization_2/ReadVariableOp:value:0;functional_1/batch_normalization_2/ReadVariableOp_1:value:0Jfunctional_1/batch_normalization_2/FusedBatchNormV3/ReadVariableOp:value:0Lfunctional_1/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ0@:@:@:@:@:*
epsilon%o:*
is_training( 25
3functional_1/batch_normalization_2/FusedBatchNormV3±
functional_1/re_lu_2/ReluRelu7functional_1/batch_normalization_2/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@2
functional_1/re_lu_2/Reluí
$functional_1/max_pooling2d_2/MaxPoolMaxPool'functional_1/re_lu_2/Relu:activations:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
ksize
*
paddingVALID*
strides
2&
$functional_1/max_pooling2d_2/MaxPoolÒ
)functional_1/conv_4/Conv2D/ReadVariableOpReadVariableOp2functional_1_conv_4_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype02+
)functional_1/conv_4/Conv2D/ReadVariableOp
functional_1/conv_4/Conv2DConv2D-functional_1/max_pooling2d_2/MaxPool:output:01functional_1/conv_4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
2
functional_1/conv_4/Conv2DÉ
*functional_1/conv_4/BiasAdd/ReadVariableOpReadVariableOp3functional_1_conv_4_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02,
*functional_1/conv_4/BiasAdd/ReadVariableOpÙ
functional_1/conv_4/BiasAddBiasAdd#functional_1/conv_4/Conv2D:output:02functional_1/conv_4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
functional_1/conv_4/BiasAddÞ
1functional_1/batch_normalization_3/ReadVariableOpReadVariableOp:functional_1_batch_normalization_3_readvariableop_resource*
_output_shapes	
:*
dtype023
1functional_1/batch_normalization_3/ReadVariableOpä
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
Dfunctional_1/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1¿
3functional_1/batch_normalization_3/FusedBatchNormV3FusedBatchNormV3$functional_1/conv_4/BiasAdd:output:09functional_1/batch_normalization_3/ReadVariableOp:value:0;functional_1/batch_normalization_3/ReadVariableOp_1:value:0Jfunctional_1/batch_normalization_3/FusedBatchNormV3/ReadVariableOp:value:0Lfunctional_1/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
is_training( 25
3functional_1/batch_normalization_3/FusedBatchNormV3²
functional_1/re_lu_3/ReluRelu7functional_1/batch_normalization_3/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
functional_1/re_lu_3/Reluî
$functional_1/max_pooling2d_3/MaxPoolMaxPool'functional_1/re_lu_3/Relu:activations:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
2&
$functional_1/max_pooling2d_3/MaxPoolÓ
)functional_1/conv_5/Conv2D/ReadVariableOpReadVariableOp2functional_1_conv_5_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02+
)functional_1/conv_5/Conv2D/ReadVariableOp
functional_1/conv_5/Conv2DConv2D-functional_1/max_pooling2d_3/MaxPool:output:01functional_1/conv_5/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
2
functional_1/conv_5/Conv2DÉ
*functional_1/conv_5/BiasAdd/ReadVariableOpReadVariableOp3functional_1_conv_5_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02,
*functional_1/conv_5/BiasAdd/ReadVariableOpÙ
functional_1/conv_5/BiasAddBiasAdd#functional_1/conv_5/Conv2D:output:02functional_1/conv_5/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
functional_1/conv_5/BiasAddÞ
1functional_1/batch_normalization_4/ReadVariableOpReadVariableOp:functional_1_batch_normalization_4_readvariableop_resource*
_output_shapes	
:*
dtype023
1functional_1/batch_normalization_4/ReadVariableOpä
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
Dfunctional_1/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1¿
3functional_1/batch_normalization_4/FusedBatchNormV3FusedBatchNormV3$functional_1/conv_5/BiasAdd:output:09functional_1/batch_normalization_4/ReadVariableOp:value:0;functional_1/batch_normalization_4/ReadVariableOp_1:value:0Jfunctional_1/batch_normalization_4/FusedBatchNormV3/ReadVariableOp:value:0Lfunctional_1/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
is_training( 25
3functional_1/batch_normalization_4/FusedBatchNormV3²
functional_1/re_lu_4/ReluRelu7functional_1/batch_normalization_4/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
functional_1/re_lu_4/Relu
functional_1/upsample_6/ShapeShape'functional_1/re_lu_4/Relu:activations:0*
T0*
_output_shapes
:2
functional_1/upsample_6/Shape¤
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
-functional_1/upsample_6/strided_slice/stack_2ò
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
functional_1/upsample_6/stack/3¢
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
/functional_1/upsample_6/strided_slice_1/stack_2ü
'functional_1/upsample_6/strided_slice_1StridedSlice&functional_1/upsample_6/stack:output:06functional_1/upsample_6/strided_slice_1/stack:output:08functional_1/upsample_6/strided_slice_1/stack_1:output:08functional_1/upsample_6/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2)
'functional_1/upsample_6/strided_slice_1ý
7functional_1/upsample_6/conv2d_transpose/ReadVariableOpReadVariableOp@functional_1_upsample_6_conv2d_transpose_readvariableop_resource*(
_output_shapes
:*
dtype029
7functional_1/upsample_6/conv2d_transpose/ReadVariableOpá
(functional_1/upsample_6/conv2d_transposeConv2DBackpropInput&functional_1/upsample_6/stack:output:0?functional_1/upsample_6/conv2d_transpose/ReadVariableOp:value:0'functional_1/re_lu_4/Relu:activations:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides
2*
(functional_1/upsample_6/conv2d_transposeÕ
.functional_1/upsample_6/BiasAdd/ReadVariableOpReadVariableOp7functional_1_upsample_6_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype020
.functional_1/upsample_6/BiasAdd/ReadVariableOpó
functional_1/upsample_6/BiasAddBiasAdd1functional_1/upsample_6/conv2d_transpose:output:06functional_1/upsample_6/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2!
functional_1/upsample_6/BiasAdd­
/functional_1/tf_op_layer_concat_6/concat_6/axisConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ21
/functional_1/tf_op_layer_concat_6/concat_6/axisÄ
*functional_1/tf_op_layer_concat_6/concat_6ConcatV2(functional_1/upsample_6/BiasAdd:output:0'functional_1/re_lu_3/Relu:activations:08functional_1/tf_op_layer_concat_6/concat_6/axis:output:0*
N*
T0*
_cloned(*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2,
*functional_1/tf_op_layer_concat_6/concat_6Ó
)functional_1/conv_6/Conv2D/ReadVariableOpReadVariableOp2functional_1_conv_6_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02+
)functional_1/conv_6/Conv2D/ReadVariableOp
functional_1/conv_6/Conv2DConv2D3functional_1/tf_op_layer_concat_6/concat_6:output:01functional_1/conv_6/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
2
functional_1/conv_6/Conv2DÉ
*functional_1/conv_6/BiasAdd/ReadVariableOpReadVariableOp3functional_1_conv_6_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02,
*functional_1/conv_6/BiasAdd/ReadVariableOpÙ
functional_1/conv_6/BiasAddBiasAdd#functional_1/conv_6/Conv2D:output:02functional_1/conv_6/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
functional_1/conv_6/BiasAddÞ
1functional_1/batch_normalization_5/ReadVariableOpReadVariableOp:functional_1_batch_normalization_5_readvariableop_resource*
_output_shapes	
:*
dtype023
1functional_1/batch_normalization_5/ReadVariableOpä
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
Dfunctional_1/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1¿
3functional_1/batch_normalization_5/FusedBatchNormV3FusedBatchNormV3$functional_1/conv_6/BiasAdd:output:09functional_1/batch_normalization_5/ReadVariableOp:value:0;functional_1/batch_normalization_5/ReadVariableOp_1:value:0Jfunctional_1/batch_normalization_5/FusedBatchNormV3/ReadVariableOp:value:0Lfunctional_1/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
is_training( 25
3functional_1/batch_normalization_5/FusedBatchNormV3²
functional_1/re_lu_5/ReluRelu7functional_1/batch_normalization_5/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
functional_1/re_lu_5/Relu
functional_1/upsample_7/ShapeShape'functional_1/re_lu_5/Relu:activations:0*
T0*
_output_shapes
:2
functional_1/upsample_7/Shape¤
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
-functional_1/upsample_7/strided_slice/stack_2ò
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
functional_1/upsample_7/stack/3¢
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
/functional_1/upsample_7/strided_slice_1/stack_2ü
'functional_1/upsample_7/strided_slice_1StridedSlice&functional_1/upsample_7/stack:output:06functional_1/upsample_7/strided_slice_1/stack:output:08functional_1/upsample_7/strided_slice_1/stack_1:output:08functional_1/upsample_7/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2)
'functional_1/upsample_7/strided_slice_1ü
7functional_1/upsample_7/conv2d_transpose/ReadVariableOpReadVariableOp@functional_1_upsample_7_conv2d_transpose_readvariableop_resource*'
_output_shapes
:@*
dtype029
7functional_1/upsample_7/conv2d_transpose/ReadVariableOpà
(functional_1/upsample_7/conv2d_transposeConv2DBackpropInput&functional_1/upsample_7/stack:output:0?functional_1/upsample_7/conv2d_transpose/ReadVariableOp:value:0'functional_1/re_lu_5/Relu:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@*
paddingVALID*
strides
2*
(functional_1/upsample_7/conv2d_transposeÔ
.functional_1/upsample_7/BiasAdd/ReadVariableOpReadVariableOp7functional_1_upsample_7_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype020
.functional_1/upsample_7/BiasAdd/ReadVariableOpò
functional_1/upsample_7/BiasAddBiasAdd1functional_1/upsample_7/conv2d_transpose:output:06functional_1/upsample_7/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@2!
functional_1/upsample_7/BiasAdd­
/functional_1/tf_op_layer_concat_7/concat_7/axisConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ21
/functional_1/tf_op_layer_concat_7/concat_7/axisÄ
*functional_1/tf_op_layer_concat_7/concat_7ConcatV2(functional_1/upsample_7/BiasAdd:output:0'functional_1/re_lu_2/Relu:activations:08functional_1/tf_op_layer_concat_7/concat_7/axis:output:0*
N*
T0*
_cloned(*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ02,
*functional_1/tf_op_layer_concat_7/concat_7Ò
)functional_1/conv_7/Conv2D/ReadVariableOpReadVariableOp2functional_1_conv_7_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype02+
)functional_1/conv_7/Conv2D/ReadVariableOp
functional_1/conv_7/Conv2DConv2D3functional_1/tf_op_layer_concat_7/concat_7:output:01functional_1/conv_7/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@*
paddingSAME*
strides
2
functional_1/conv_7/Conv2DÈ
*functional_1/conv_7/BiasAdd/ReadVariableOpReadVariableOp3functional_1_conv_7_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02,
*functional_1/conv_7/BiasAdd/ReadVariableOpØ
functional_1/conv_7/BiasAddBiasAdd#functional_1/conv_7/Conv2D:output:02functional_1/conv_7/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@2
functional_1/conv_7/BiasAddÝ
1functional_1/batch_normalization_6/ReadVariableOpReadVariableOp:functional_1_batch_normalization_6_readvariableop_resource*
_output_shapes
:@*
dtype023
1functional_1/batch_normalization_6/ReadVariableOpã
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
Dfunctional_1/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1º
3functional_1/batch_normalization_6/FusedBatchNormV3FusedBatchNormV3$functional_1/conv_7/BiasAdd:output:09functional_1/batch_normalization_6/ReadVariableOp:value:0;functional_1/batch_normalization_6/ReadVariableOp_1:value:0Jfunctional_1/batch_normalization_6/FusedBatchNormV3/ReadVariableOp:value:0Lfunctional_1/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ0@:@:@:@:@:*
epsilon%o:*
is_training( 25
3functional_1/batch_normalization_6/FusedBatchNormV3±
functional_1/re_lu_6/ReluRelu7functional_1/batch_normalization_6/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@2
functional_1/re_lu_6/Relu
functional_1/upsample_8/ShapeShape'functional_1/re_lu_6/Relu:activations:0*
T0*
_output_shapes
:2
functional_1/upsample_8/Shape¤
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
-functional_1/upsample_8/strided_slice/stack_2ò
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
functional_1/upsample_8/stack/3¢
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
/functional_1/upsample_8/strided_slice_1/stack_2ü
'functional_1/upsample_8/strided_slice_1StridedSlice&functional_1/upsample_8/stack:output:06functional_1/upsample_8/strided_slice_1/stack:output:08functional_1/upsample_8/strided_slice_1/stack_1:output:08functional_1/upsample_8/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2)
'functional_1/upsample_8/strided_slice_1û
7functional_1/upsample_8/conv2d_transpose/ReadVariableOpReadVariableOp@functional_1_upsample_8_conv2d_transpose_readvariableop_resource*&
_output_shapes
: @*
dtype029
7functional_1/upsample_8/conv2d_transpose/ReadVariableOpà
(functional_1/upsample_8/conv2d_transposeConv2DBackpropInput&functional_1/upsample_8/stack:output:0?functional_1/upsample_8/conv2d_transpose/ReadVariableOp:value:0'functional_1/re_lu_6/Relu:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0` *
paddingVALID*
strides
2*
(functional_1/upsample_8/conv2d_transposeÔ
.functional_1/upsample_8/BiasAdd/ReadVariableOpReadVariableOp7functional_1_upsample_8_biasadd_readvariableop_resource*
_output_shapes
: *
dtype020
.functional_1/upsample_8/BiasAdd/ReadVariableOpò
functional_1/upsample_8/BiasAddBiasAdd1functional_1/upsample_8/conv2d_transpose:output:06functional_1/upsample_8/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0` 2!
functional_1/upsample_8/BiasAdd­
/functional_1/tf_op_layer_concat_8/concat_8/axisConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ21
/functional_1/tf_op_layer_concat_8/concat_8/axisÃ
*functional_1/tf_op_layer_concat_8/concat_8ConcatV2(functional_1/upsample_8/BiasAdd:output:0'functional_1/re_lu_1/Relu:activations:08functional_1/tf_op_layer_concat_8/concat_8/axis:output:0*
N*
T0*
_cloned(*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0`@2,
*functional_1/tf_op_layer_concat_8/concat_8Ñ
)functional_1/conv_8/Conv2D/ReadVariableOpReadVariableOp2functional_1_conv_8_conv2d_readvariableop_resource*&
_output_shapes
:@ *
dtype02+
)functional_1/conv_8/Conv2D/ReadVariableOp
functional_1/conv_8/Conv2DConv2D3functional_1/tf_op_layer_concat_8/concat_8:output:01functional_1/conv_8/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0` *
paddingSAME*
strides
2
functional_1/conv_8/Conv2DÈ
*functional_1/conv_8/BiasAdd/ReadVariableOpReadVariableOp3functional_1_conv_8_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02,
*functional_1/conv_8/BiasAdd/ReadVariableOpØ
functional_1/conv_8/BiasAddBiasAdd#functional_1/conv_8/Conv2D:output:02functional_1/conv_8/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0` 2
functional_1/conv_8/BiasAddÝ
1functional_1/batch_normalization_7/ReadVariableOpReadVariableOp:functional_1_batch_normalization_7_readvariableop_resource*
_output_shapes
: *
dtype023
1functional_1/batch_normalization_7/ReadVariableOpã
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
Dfunctional_1/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1º
3functional_1/batch_normalization_7/FusedBatchNormV3FusedBatchNormV3$functional_1/conv_8/BiasAdd:output:09functional_1/batch_normalization_7/ReadVariableOp:value:0;functional_1/batch_normalization_7/ReadVariableOp_1:value:0Jfunctional_1/batch_normalization_7/FusedBatchNormV3/ReadVariableOp:value:0Lfunctional_1/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ0` : : : : :*
epsilon%o:*
is_training( 25
3functional_1/batch_normalization_7/FusedBatchNormV3±
functional_1/re_lu_7/ReluRelu7functional_1/batch_normalization_7/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0` 2
functional_1/re_lu_7/Relu
functional_1/upsample_9/ShapeShape'functional_1/re_lu_7/Relu:activations:0*
T0*
_output_shapes
:2
functional_1/upsample_9/Shape¤
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
-functional_1/upsample_9/strided_slice/stack_2ò
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
B :À2!
functional_1/upsample_9/stack/2
functional_1/upsample_9/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2!
functional_1/upsample_9/stack/3¢
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
/functional_1/upsample_9/strided_slice_1/stack_2ü
'functional_1/upsample_9/strided_slice_1StridedSlice&functional_1/upsample_9/stack:output:06functional_1/upsample_9/strided_slice_1/stack:output:08functional_1/upsample_9/strided_slice_1/stack_1:output:08functional_1/upsample_9/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2)
'functional_1/upsample_9/strided_slice_1û
7functional_1/upsample_9/conv2d_transpose/ReadVariableOpReadVariableOp@functional_1_upsample_9_conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype029
7functional_1/upsample_9/conv2d_transpose/ReadVariableOpá
(functional_1/upsample_9/conv2d_transposeConv2DBackpropInput&functional_1/upsample_9/stack:output:0?functional_1/upsample_9/conv2d_transpose/ReadVariableOp:value:0'functional_1/re_lu_7/Relu:activations:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`À*
paddingVALID*
strides
2*
(functional_1/upsample_9/conv2d_transposeÔ
.functional_1/upsample_9/BiasAdd/ReadVariableOpReadVariableOp7functional_1_upsample_9_biasadd_readvariableop_resource*
_output_shapes
:*
dtype020
.functional_1/upsample_9/BiasAdd/ReadVariableOpó
functional_1/upsample_9/BiasAddBiasAdd1functional_1/upsample_9/conv2d_transpose:output:06functional_1/upsample_9/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`À2!
functional_1/upsample_9/BiasAdd­
/functional_1/tf_op_layer_concat_9/concat_9/axisConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ21
/functional_1/tf_op_layer_concat_9/concat_9/axisÂ
*functional_1/tf_op_layer_concat_9/concat_9ConcatV2(functional_1/upsample_9/BiasAdd:output:0%functional_1/re_lu/Relu:activations:08functional_1/tf_op_layer_concat_9/concat_9/axis:output:0*
N*
T0*
_cloned(*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`À 2,
*functional_1/tf_op_layer_concat_9/concat_9Ñ
)functional_1/conv_9/Conv2D/ReadVariableOpReadVariableOp2functional_1_conv_9_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02+
)functional_1/conv_9/Conv2D/ReadVariableOp
functional_1/conv_9/Conv2DConv2D3functional_1/tf_op_layer_concat_9/concat_9:output:01functional_1/conv_9/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`À*
paddingSAME*
strides
2
functional_1/conv_9/Conv2DÈ
*functional_1/conv_9/BiasAdd/ReadVariableOpReadVariableOp3functional_1_conv_9_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02,
*functional_1/conv_9/BiasAdd/ReadVariableOpÙ
functional_1/conv_9/BiasAddBiasAdd#functional_1/conv_9/Conv2D:output:02functional_1/conv_9/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`À2
functional_1/conv_9/BiasAddÝ
1functional_1/batch_normalization_8/ReadVariableOpReadVariableOp:functional_1_batch_normalization_8_readvariableop_resource*
_output_shapes
:*
dtype023
1functional_1/batch_normalization_8/ReadVariableOpã
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
8:ÿÿÿÿÿÿÿÿÿ`À:::::*
epsilon%o:*
is_training( 25
3functional_1/batch_normalization_8/FusedBatchNormV3²
functional_1/re_lu_8/ReluRelu7functional_1/batch_normalization_8/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`À2
functional_1/re_lu_8/ReluÎ
(functional_1/final/Conv2D/ReadVariableOpReadVariableOp1functional_1_final_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02*
(functional_1/final/Conv2D/ReadVariableOpþ
functional_1/final/Conv2DConv2D'functional_1/re_lu_8/Relu:activations:00functional_1/final/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`À*
paddingSAME*
strides
2
functional_1/final/Conv2DÅ
)functional_1/final/BiasAdd/ReadVariableOpReadVariableOp2functional_1_final_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02+
)functional_1/final/BiasAdd/ReadVariableOpÕ
functional_1/final/BiasAddBiasAdd"functional_1/final/Conv2D:output:01functional_1/final/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`À2
functional_1/final/BiasAdd
IdentityIdentity#functional_1/final/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`À2

Identity"
identityIdentity:output:0*±
_input_shapes
:ÿÿÿÿÿÿÿÿÿ`À:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::\ X
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`À
$
_user_specified_name
imageInput
Õ

R__inference_batch_normalization_layer_call_and_return_conditional_losses_107948063

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
!FusedBatchNormV3/ReadVariableOp_1Ë
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*L
_output_shapes:
8:ÿÿÿÿÿÿÿÿÿ`À:::::*
epsilon%o:*
is_training( 2
FusedBatchNormV3q
IdentityIdentityFusedBatchNormV3:y:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`À2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿ`À:::::X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`À
 
_user_specified_nameinputs
Ö
`
D__inference_re_lu_layer_call_and_return_conditional_losses_107945111

inputs
identityW
ReluReluinputs*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`À2
Reluo
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`À2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ`À:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`À
 
_user_specified_nameinputs
Ø
±
T__inference_batch_normalization_5_layer_call_and_return_conditional_losses_107948779

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity¢AssignNewValue¢AssignNewValue_1u
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
FusedBatchNormV3/ReadVariableOp®
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ï
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3ÿ
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
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
®
­
E__inference_conv_6_layer_call_and_return_conditional_losses_107948750

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp¤
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
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
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddm
IdentityIdentityBiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:ÿÿÿÿÿÿÿÿÿ:::X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ß

T__inference_batch_normalization_3_layer_call_and_return_conditional_losses_107945409

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
FusedBatchNormV3/ReadVariableOp®
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ï
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
is_training( 2
FusedBatchNormV3q
IdentityIdentityFusedBatchNormV3:y:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿ:::::X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
®
¬
9__inference_batch_normalization_5_layer_call_fn_107948823

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall¾
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *]
fXRV
T__inference_batch_normalization_5_layer_call_and_return_conditional_losses_1079445082
StatefulPartitionedCall©
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ::::22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¨
¬
9__inference_batch_normalization_8_layer_call_fn_107949320

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall»
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *]
fXRV
T__inference_batch_normalization_8_layer_call_and_return_conditional_losses_1079449332
StatefulPartitionedCall¨
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
â
¬
9__inference_batch_normalization_6_layer_call_fn_107949057

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall«
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *]
fXRV
T__inference_batch_normalization_6_layer_call_and_return_conditional_losses_1079457882
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ0@::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@
 
_user_specified_nameinputs
¸
}
S__inference_tf_op_layer_concat_8_layer_call_and_return_conditional_losses_107945849

inputs
inputs_1
identityi
concat_8/axisConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
concat_8/axis
concat_8ConcatV2inputsinputs_1concat_8/axis:output:0*
N*
T0*
_cloned(*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0`@2

concat_8m
IdentityIdentityconcat_8:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0`@2

Identity"
identityIdentity:output:0*[
_input_shapesJ
H:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ0` :i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs:WS
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0` 
 
_user_specified_nameinputs


T__inference_batch_normalization_8_layer_call_and_return_conditional_losses_107949307

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
!FusedBatchNormV3/ReadVariableOp_1Ü
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
is_training( 2
FusedBatchNormV3
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


T__inference_batch_normalization_6_layer_call_and_return_conditional_losses_107944660

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
!FusedBatchNormV3/ReadVariableOp_1Ü
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@:@:@:@:@:*
epsilon%o:*
is_training( 2
FusedBatchNormV3
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@:::::i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
ª
¬
9__inference_batch_normalization_8_layer_call_fn_107949333

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall½
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *]
fXRV
T__inference_batch_normalization_8_layer_call_and_return_conditional_losses_1079449642
StatefulPartitionedCall¨
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¦
­
E__inference_conv_2_layer_call_and_return_conditional_losses_107948109

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
:ÿÿÿÿÿÿÿÿÿ0` *
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
:ÿÿÿÿÿÿÿÿÿ0` 2	
BiasAddl
IdentityIdentityBiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0` 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ0`:::W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0`
 
_user_specified_nameinputs
´
M
1__inference_max_pooling2d_layer_call_fn_107943915

inputs
identityò
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *U
fPRN
L__inference_max_pooling2d_layer_call_and_return_conditional_losses_1079439092
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

±
T__inference_batch_normalization_8_layer_call_and_return_conditional_losses_107946036

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity¢AssignNewValue¢AssignNewValue_1t
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
!FusedBatchNormV3/ReadVariableOp_1Ù
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*L
_output_shapes:
8:ÿÿÿÿÿÿÿÿÿ`À:::::*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3ÿ
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
:ÿÿÿÿÿÿÿÿÿ`À2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿ`À::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`À
 
_user_specified_nameinputs
«
­
E__inference_conv_9_layer_call_and_return_conditional_losses_107949260

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOp¤
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`À*
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
:ÿÿÿÿÿÿÿÿÿ`À2	
BiasAddm
IdentityIdentityBiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`À2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:ÿÿÿÿÿÿÿÿÿ`À :::X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`À 
 
_user_specified_nameinputs


*__inference_conv_9_layer_call_fn_107949269

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`À*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_conv_9_layer_call_and_return_conditional_losses_1079460012
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`À2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:ÿÿÿÿÿÿÿÿÿ`À ::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`À 
 
_user_specified_nameinputs
Ø
±
T__inference_batch_normalization_4_layer_call_and_return_conditional_losses_107944325

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity¢AssignNewValue¢AssignNewValue_1u
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
FusedBatchNormV3/ReadVariableOp®
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ï
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3ÿ
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
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
º
}
S__inference_tf_op_layer_concat_7_layer_call_and_return_conditional_losses_107945716

inputs
inputs_1
identityi
concat_7/axisConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
concat_7/axis
concat_7ConcatV2inputsinputs_1concat_7/axis:output:0*
N*
T0*
_cloned(*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ02

concat_7n
IdentityIdentityconcat_7:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ02

Identity"
identityIdentity:output:0*[
_input_shapesJ
H:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ0@:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs:WS
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@
 
_user_specified_nameinputs
Ø
b
F__inference_re_lu_8_layer_call_and_return_conditional_losses_107946095

inputs
identityW
ReluReluinputs*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`À2
Reluo
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`À2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ`À:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`À
 
_user_specified_nameinputs
®
­
E__inference_conv_5_layer_call_and_return_conditional_losses_107945469

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp¤
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
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
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddm
IdentityIdentityBiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:ÿÿÿÿÿÿÿÿÿ:::X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ª
¬
D__inference_final_layer_call_and_return_conditional_losses_107946113

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp¤
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`À*
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
:ÿÿÿÿÿÿÿÿÿ`À2	
BiasAddm
IdentityIdentityBiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`À2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:ÿÿÿÿÿÿÿÿÿ`À:::X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`À
 
_user_specified_nameinputs

¯
R__inference_batch_normalization_layer_call_and_return_conditional_losses_107948045

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity¢AssignNewValue¢AssignNewValue_1t
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
!FusedBatchNormV3/ReadVariableOp_1Ù
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*L
_output_shapes:
8:ÿÿÿÿÿÿÿÿÿ`À:::::*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3ÿ
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
:ÿÿÿÿÿÿÿÿÿ`À2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿ`À::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`À
 
_user_specified_nameinputs
ª
¬
9__inference_batch_normalization_2_layer_call_fn_107948339

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall½
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *]
fXRV
T__inference_batch_normalization_2_layer_call_and_return_conditional_losses_1079441242
StatefulPartitionedCall¨
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
Ó

T__inference_batch_normalization_6_layer_call_and_return_conditional_losses_107945788

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
!FusedBatchNormV3/ReadVariableOp_1Ê
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ0@:@:@:@:@:*
epsilon%o:*
is_training( 2
FusedBatchNormV3p
IdentityIdentityFusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ0@:::::W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@
 
_user_specified_nameinputs
à
¬
9__inference_batch_normalization_2_layer_call_fn_107948390

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall©
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *]
fXRV
T__inference_batch_normalization_2_layer_call_and_return_conditional_losses_1079452782
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ0@::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@
 
_user_specified_nameinputs
¨

T__inference_batch_normalization_3_layer_call_and_return_conditional_losses_107944240

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
FusedBatchNormV3/ReadVariableOp®
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1á
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
is_training( 2
FusedBatchNormV3
IdentityIdentityFusedBatchNormV3:y:0*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
æ
¬
9__inference_batch_normalization_8_layer_call_fn_107949397

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall¬
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`À*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *]
fXRV
T__inference_batch_normalization_8_layer_call_and_return_conditional_losses_1079460542
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`À2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿ`À::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`À
 
_user_specified_nameinputs

±
T__inference_batch_normalization_6_layer_call_and_return_conditional_losses_107945770

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity¢AssignNewValue¢AssignNewValue_1t
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
!FusedBatchNormV3/ReadVariableOp_1Ø
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ0@:@:@:@:@:*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3ÿ
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
:ÿÿÿÿÿÿÿÿÿ0@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ0@::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@
 
_user_specified_nameinputs

±
T__inference_batch_normalization_4_layer_call_and_return_conditional_losses_107948673

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity¢AssignNewValue¢AssignNewValue_1u
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
FusedBatchNormV3/ReadVariableOp®
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ý
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3ÿ
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
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¥Ë
§
K__inference_functional_1_layer_call_and_return_conditional_losses_107946481

inputs
conv_1_107946312
conv_1_107946314!
batch_normalization_107946317!
batch_normalization_107946319!
batch_normalization_107946321!
batch_normalization_107946323
conv_2_107946328
conv_2_107946330#
batch_normalization_1_107946333#
batch_normalization_1_107946335#
batch_normalization_1_107946337#
batch_normalization_1_107946339
conv_3_107946344
conv_3_107946346#
batch_normalization_2_107946349#
batch_normalization_2_107946351#
batch_normalization_2_107946353#
batch_normalization_2_107946355
conv_4_107946360
conv_4_107946362#
batch_normalization_3_107946365#
batch_normalization_3_107946367#
batch_normalization_3_107946369#
batch_normalization_3_107946371
conv_5_107946376
conv_5_107946378#
batch_normalization_4_107946381#
batch_normalization_4_107946383#
batch_normalization_4_107946385#
batch_normalization_4_107946387
upsample_6_107946391
upsample_6_107946393
conv_6_107946397
conv_6_107946399#
batch_normalization_5_107946402#
batch_normalization_5_107946404#
batch_normalization_5_107946406#
batch_normalization_5_107946408
upsample_7_107946412
upsample_7_107946414
conv_7_107946418
conv_7_107946420#
batch_normalization_6_107946423#
batch_normalization_6_107946425#
batch_normalization_6_107946427#
batch_normalization_6_107946429
upsample_8_107946433
upsample_8_107946435
conv_8_107946439
conv_8_107946441#
batch_normalization_7_107946444#
batch_normalization_7_107946446#
batch_normalization_7_107946448#
batch_normalization_7_107946450
upsample_9_107946454
upsample_9_107946456
conv_9_107946460
conv_9_107946462#
batch_normalization_8_107946465#
batch_normalization_8_107946467#
batch_normalization_8_107946469#
batch_normalization_8_107946471
final_107946475
final_107946477
identity¢+batch_normalization/StatefulPartitionedCall¢-batch_normalization_1/StatefulPartitionedCall¢-batch_normalization_2/StatefulPartitionedCall¢-batch_normalization_3/StatefulPartitionedCall¢-batch_normalization_4/StatefulPartitionedCall¢-batch_normalization_5/StatefulPartitionedCall¢-batch_normalization_6/StatefulPartitionedCall¢-batch_normalization_7/StatefulPartitionedCall¢-batch_normalization_8/StatefulPartitionedCall¢conv_1/StatefulPartitionedCall¢conv_2/StatefulPartitionedCall¢conv_3/StatefulPartitionedCall¢conv_4/StatefulPartitionedCall¢conv_5/StatefulPartitionedCall¢conv_6/StatefulPartitionedCall¢conv_7/StatefulPartitionedCall¢conv_8/StatefulPartitionedCall¢conv_9/StatefulPartitionedCall¢final/StatefulPartitionedCall¢"upsample_6/StatefulPartitionedCall¢"upsample_7/StatefulPartitionedCall¢"upsample_8/StatefulPartitionedCall¢"upsample_9/StatefulPartitionedCall
#tf_op_layer_RealDiv/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`À* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *[
fVRT
R__inference_tf_op_layer_RealDiv_layer_call_and_return_conditional_losses_1079449852%
#tf_op_layer_RealDiv/PartitionedCall 
tf_op_layer_Sub/PartitionedCallPartitionedCall,tf_op_layer_RealDiv/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`À* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *W
fRRP
N__inference_tf_op_layer_Sub_layer_call_and_return_conditional_losses_1079449992!
tf_op_layer_Sub/PartitionedCallÃ
conv_1/StatefulPartitionedCallStatefulPartitionedCall(tf_op_layer_Sub/PartitionedCall:output:0conv_1_107946312conv_1_107946314*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`À*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_conv_1_layer_call_and_return_conditional_losses_1079450172 
conv_1/StatefulPartitionedCallÃ
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall'conv_1/StatefulPartitionedCall:output:0batch_normalization_107946317batch_normalization_107946319batch_normalization_107946321batch_normalization_107946323*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`À*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *[
fVRT
R__inference_batch_normalization_layer_call_and_return_conditional_losses_1079450522-
+batch_normalization/StatefulPartitionedCall
re_lu/PartitionedCallPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`À* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *M
fHRF
D__inference_re_lu_layer_call_and_return_conditional_losses_1079451112
re_lu/PartitionedCall
max_pooling2d/PartitionedCallPartitionedCallre_lu/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0`* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *U
fPRN
L__inference_max_pooling2d_layer_call_and_return_conditional_losses_1079439092
max_pooling2d/PartitionedCallÀ
conv_2/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0conv_2_107946328conv_2_107946330*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0` *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_conv_2_layer_call_and_return_conditional_losses_1079451302 
conv_2/StatefulPartitionedCallÐ
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall'conv_2/StatefulPartitionedCall:output:0batch_normalization_1_107946333batch_normalization_1_107946335batch_normalization_1_107946337batch_normalization_1_107946339*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0` *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *]
fXRV
T__inference_batch_normalization_1_layer_call_and_return_conditional_losses_1079451652/
-batch_normalization_1/StatefulPartitionedCall
re_lu_1/PartitionedCallPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0` * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *O
fJRH
F__inference_re_lu_1_layer_call_and_return_conditional_losses_1079452242
re_lu_1/PartitionedCall
max_pooling2d_1/PartitionedCallPartitionedCall re_lu_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0 * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *W
fRRP
N__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_1079440252!
max_pooling2d_1/PartitionedCallÂ
conv_3/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_1/PartitionedCall:output:0conv_3_107946344conv_3_107946346*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_conv_3_layer_call_and_return_conditional_losses_1079452432 
conv_3/StatefulPartitionedCallÐ
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall'conv_3/StatefulPartitionedCall:output:0batch_normalization_2_107946349batch_normalization_2_107946351batch_normalization_2_107946353batch_normalization_2_107946355*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *]
fXRV
T__inference_batch_normalization_2_layer_call_and_return_conditional_losses_1079452782/
-batch_normalization_2/StatefulPartitionedCall
re_lu_2/PartitionedCallPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *O
fJRH
F__inference_re_lu_2_layer_call_and_return_conditional_losses_1079453372
re_lu_2/PartitionedCall
max_pooling2d_2/PartitionedCallPartitionedCall re_lu_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *W
fRRP
N__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_1079441412!
max_pooling2d_2/PartitionedCallÃ
conv_4/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_2/PartitionedCall:output:0conv_4_107946360conv_4_107946362*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_conv_4_layer_call_and_return_conditional_losses_1079453562 
conv_4/StatefulPartitionedCallÑ
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCall'conv_4/StatefulPartitionedCall:output:0batch_normalization_3_107946365batch_normalization_3_107946367batch_normalization_3_107946369batch_normalization_3_107946371*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *]
fXRV
T__inference_batch_normalization_3_layer_call_and_return_conditional_losses_1079453912/
-batch_normalization_3/StatefulPartitionedCall
re_lu_3/PartitionedCallPartitionedCall6batch_normalization_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *O
fJRH
F__inference_re_lu_3_layer_call_and_return_conditional_losses_1079454502
re_lu_3/PartitionedCall
max_pooling2d_3/PartitionedCallPartitionedCall re_lu_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *W
fRRP
N__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_1079442572!
max_pooling2d_3/PartitionedCallÃ
conv_5/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_3/PartitionedCall:output:0conv_5_107946376conv_5_107946378*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_conv_5_layer_call_and_return_conditional_losses_1079454692 
conv_5/StatefulPartitionedCallÑ
-batch_normalization_4/StatefulPartitionedCallStatefulPartitionedCall'conv_5/StatefulPartitionedCall:output:0batch_normalization_4_107946381batch_normalization_4_107946383batch_normalization_4_107946385batch_normalization_4_107946387*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *]
fXRV
T__inference_batch_normalization_4_layer_call_and_return_conditional_losses_1079455042/
-batch_normalization_4/StatefulPartitionedCall
re_lu_4/PartitionedCallPartitionedCall6batch_normalization_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *O
fJRH
F__inference_re_lu_4_layer_call_and_return_conditional_losses_1079455632
re_lu_4/PartitionedCallá
"upsample_6/StatefulPartitionedCallStatefulPartitionedCall re_lu_4/PartitionedCall:output:0upsample_6_107946391upsample_6_107946393*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *R
fMRK
I__inference_upsample_6_layer_call_and_return_conditional_losses_1079444052$
"upsample_6/StatefulPartitionedCallÑ
$tf_op_layer_concat_6/PartitionedCallPartitionedCall+upsample_6/StatefulPartitionedCall:output:0 re_lu_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *\
fWRU
S__inference_tf_op_layer_concat_6_layer_call_and_return_conditional_losses_1079455832&
$tf_op_layer_concat_6/PartitionedCallÈ
conv_6/StatefulPartitionedCallStatefulPartitionedCall-tf_op_layer_concat_6/PartitionedCall:output:0conv_6_107946397conv_6_107946399*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_conv_6_layer_call_and_return_conditional_losses_1079456022 
conv_6/StatefulPartitionedCallÑ
-batch_normalization_5/StatefulPartitionedCallStatefulPartitionedCall'conv_6/StatefulPartitionedCall:output:0batch_normalization_5_107946402batch_normalization_5_107946404batch_normalization_5_107946406batch_normalization_5_107946408*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *]
fXRV
T__inference_batch_normalization_5_layer_call_and_return_conditional_losses_1079456372/
-batch_normalization_5/StatefulPartitionedCall
re_lu_5/PartitionedCallPartitionedCall6batch_normalization_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *O
fJRH
F__inference_re_lu_5_layer_call_and_return_conditional_losses_1079456962
re_lu_5/PartitionedCallà
"upsample_7/StatefulPartitionedCallStatefulPartitionedCall re_lu_5/PartitionedCall:output:0upsample_7_107946412upsample_7_107946414*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *R
fMRK
I__inference_upsample_7_layer_call_and_return_conditional_losses_1079445572$
"upsample_7/StatefulPartitionedCallÑ
$tf_op_layer_concat_7/PartitionedCallPartitionedCall+upsample_7/StatefulPartitionedCall:output:0 re_lu_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *\
fWRU
S__inference_tf_op_layer_concat_7_layer_call_and_return_conditional_losses_1079457162&
$tf_op_layer_concat_7/PartitionedCallÇ
conv_7/StatefulPartitionedCallStatefulPartitionedCall-tf_op_layer_concat_7/PartitionedCall:output:0conv_7_107946418conv_7_107946420*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_conv_7_layer_call_and_return_conditional_losses_1079457352 
conv_7/StatefulPartitionedCallÐ
-batch_normalization_6/StatefulPartitionedCallStatefulPartitionedCall'conv_7/StatefulPartitionedCall:output:0batch_normalization_6_107946423batch_normalization_6_107946425batch_normalization_6_107946427batch_normalization_6_107946429*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *]
fXRV
T__inference_batch_normalization_6_layer_call_and_return_conditional_losses_1079457702/
-batch_normalization_6/StatefulPartitionedCall
re_lu_6/PartitionedCallPartitionedCall6batch_normalization_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *O
fJRH
F__inference_re_lu_6_layer_call_and_return_conditional_losses_1079458292
re_lu_6/PartitionedCallà
"upsample_8/StatefulPartitionedCallStatefulPartitionedCall re_lu_6/PartitionedCall:output:0upsample_8_107946433upsample_8_107946435*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *R
fMRK
I__inference_upsample_8_layer_call_and_return_conditional_losses_1079447092$
"upsample_8/StatefulPartitionedCallÐ
$tf_op_layer_concat_8/PartitionedCallPartitionedCall+upsample_8/StatefulPartitionedCall:output:0 re_lu_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0`@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *\
fWRU
S__inference_tf_op_layer_concat_8_layer_call_and_return_conditional_losses_1079458492&
$tf_op_layer_concat_8/PartitionedCallÇ
conv_8/StatefulPartitionedCallStatefulPartitionedCall-tf_op_layer_concat_8/PartitionedCall:output:0conv_8_107946439conv_8_107946441*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0` *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_conv_8_layer_call_and_return_conditional_losses_1079458682 
conv_8/StatefulPartitionedCallÐ
-batch_normalization_7/StatefulPartitionedCallStatefulPartitionedCall'conv_8/StatefulPartitionedCall:output:0batch_normalization_7_107946444batch_normalization_7_107946446batch_normalization_7_107946448batch_normalization_7_107946450*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0` *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *]
fXRV
T__inference_batch_normalization_7_layer_call_and_return_conditional_losses_1079459032/
-batch_normalization_7/StatefulPartitionedCall
re_lu_7/PartitionedCallPartitionedCall6batch_normalization_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0` * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *O
fJRH
F__inference_re_lu_7_layer_call_and_return_conditional_losses_1079459622
re_lu_7/PartitionedCallà
"upsample_9/StatefulPartitionedCallStatefulPartitionedCall re_lu_7/PartitionedCall:output:0upsample_9_107946454upsample_9_107946456*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *R
fMRK
I__inference_upsample_9_layer_call_and_return_conditional_losses_1079448612$
"upsample_9/StatefulPartitionedCallÏ
$tf_op_layer_concat_9/PartitionedCallPartitionedCall+upsample_9/StatefulPartitionedCall:output:0re_lu/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`À * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *\
fWRU
S__inference_tf_op_layer_concat_9_layer_call_and_return_conditional_losses_1079459822&
$tf_op_layer_concat_9/PartitionedCallÈ
conv_9/StatefulPartitionedCallStatefulPartitionedCall-tf_op_layer_concat_9/PartitionedCall:output:0conv_9_107946460conv_9_107946462*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`À*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_conv_9_layer_call_and_return_conditional_losses_1079460012 
conv_9/StatefulPartitionedCallÑ
-batch_normalization_8/StatefulPartitionedCallStatefulPartitionedCall'conv_9/StatefulPartitionedCall:output:0batch_normalization_8_107946465batch_normalization_8_107946467batch_normalization_8_107946469batch_normalization_8_107946471*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`À*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *]
fXRV
T__inference_batch_normalization_8_layer_call_and_return_conditional_losses_1079460362/
-batch_normalization_8/StatefulPartitionedCall
re_lu_8/PartitionedCallPartitionedCall6batch_normalization_8/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`À* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *O
fJRH
F__inference_re_lu_8_layer_call_and_return_conditional_losses_1079460952
re_lu_8/PartitionedCall¶
final/StatefulPartitionedCallStatefulPartitionedCall re_lu_8/PartitionedCall:output:0final_107946475final_107946477*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`À*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *M
fHRF
D__inference_final_layer_call_and_return_conditional_losses_1079461132
final/StatefulPartitionedCall
IdentityIdentity&final/StatefulPartitionedCall:output:0,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall.^batch_normalization_3/StatefulPartitionedCall.^batch_normalization_4/StatefulPartitionedCall.^batch_normalization_5/StatefulPartitionedCall.^batch_normalization_6/StatefulPartitionedCall.^batch_normalization_7/StatefulPartitionedCall.^batch_normalization_8/StatefulPartitionedCall^conv_1/StatefulPartitionedCall^conv_2/StatefulPartitionedCall^conv_3/StatefulPartitionedCall^conv_4/StatefulPartitionedCall^conv_5/StatefulPartitionedCall^conv_6/StatefulPartitionedCall^conv_7/StatefulPartitionedCall^conv_8/StatefulPartitionedCall^conv_9/StatefulPartitionedCall^final/StatefulPartitionedCall#^upsample_6/StatefulPartitionedCall#^upsample_7/StatefulPartitionedCall#^upsample_8/StatefulPartitionedCall#^upsample_9/StatefulPartitionedCall*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`À2

Identity"
identityIdentity:output:0*±
_input_shapes
:ÿÿÿÿÿÿÿÿÿ`À::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::2Z
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
:ÿÿÿÿÿÿÿÿÿ`À
 
_user_specified_nameinputs
Ì
±
T__inference_batch_normalization_6_layer_call_and_return_conditional_losses_107944629

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity¢AssignNewValue¢AssignNewValue_1t
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
!FusedBatchNormV3/ReadVariableOp_1ê
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@:@:@:@:@:*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3ÿ
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
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
â
¬
9__inference_batch_normalization_1_layer_call_fn_107948246

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall«
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0` *&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *]
fXRV
T__inference_batch_normalization_1_layer_call_and_return_conditional_losses_1079451832
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0` 2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ0` ::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0` 
 
_user_specified_nameinputs
Ô
b
F__inference_re_lu_1_layer_call_and_return_conditional_losses_107948251

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0` 2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0` 2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ0` :W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0` 
 
_user_specified_nameinputs
Ø
b
F__inference_re_lu_8_layer_call_and_return_conditional_losses_107949402

inputs
identityW
ReluReluinputs*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`À2
Reluo
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`À2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ`À:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`À
 
_user_specified_nameinputs
¬
¬
9__inference_batch_normalization_5_layer_call_fn_107948810

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall¼
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *]
fXRV
T__inference_batch_normalization_5_layer_call_and_return_conditional_losses_1079444772
StatefulPartitionedCall©
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ::::22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¾
}
S__inference_tf_op_layer_concat_6_layer_call_and_return_conditional_losses_107945583

inputs
inputs_1
identityi
concat_6/axisConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
concat_6/axis
concat_6ConcatV2inputsinputs_1concat_6/axis:output:0*
N*
T0*
_cloned(*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

concat_6n
IdentityIdentityconcat_6:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*]
_input_shapesL
J:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:XT
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¨

T__inference_batch_normalization_4_layer_call_and_return_conditional_losses_107948627

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
FusedBatchNormV3/ReadVariableOp®
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1á
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
is_training( 2
FusedBatchNormV3
IdentityIdentityFusedBatchNormV3:y:0*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


T__inference_batch_normalization_1_layer_call_and_return_conditional_losses_107948156

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
!FusedBatchNormV3/ReadVariableOp_1Ü
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ : : : : :*
epsilon%o:*
is_training( 2
FusedBatchNormV3
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ :::::i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs


T__inference_batch_normalization_7_layer_call_and_return_conditional_losses_107949137

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
!FusedBatchNormV3/ReadVariableOp_1Ü
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ : : : : :*
epsilon%o:*
is_training( 2
FusedBatchNormV3
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ :::::i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
ß

T__inference_batch_normalization_5_layer_call_and_return_conditional_losses_107945655

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
FusedBatchNormV3/ReadVariableOp®
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ï
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
is_training( 2
FusedBatchNormV3q
IdentityIdentityFusedBatchNormV3:y:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿ:::::X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

j
N__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_107944141

inputs
identity­
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
2	
MaxPool
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
æ
¬
9__inference_batch_normalization_5_layer_call_fn_107948887

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall¬
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *]
fXRV
T__inference_batch_normalization_5_layer_call_and_return_conditional_losses_1079456552
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿ::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¼
}
S__inference_tf_op_layer_concat_9_layer_call_and_return_conditional_losses_107945982

inputs
inputs_1
identityi
concat_9/axisConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
concat_9/axis
concat_9ConcatV2inputsinputs_1concat_9/axis:output:0*
N*
T0*
_cloned(*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`À 2

concat_9n
IdentityIdentityconcat_9:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`À 2

Identity"
identityIdentity:output:0*\
_input_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ`À:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:XT
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`À
 
_user_specified_nameinputs
ä
¬
9__inference_batch_normalization_8_layer_call_fn_107949384

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCallª
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`À*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *]
fXRV
T__inference_batch_normalization_8_layer_call_and_return_conditional_losses_1079460362
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`À2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿ`À::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`À
 
_user_specified_nameinputs
¨
¬
9__inference_batch_normalization_1_layer_call_fn_107948169

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall»
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *]
fXRV
T__inference_batch_normalization_1_layer_call_and_return_conditional_losses_1079439772
StatefulPartitionedCall¨
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ ::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
Ó

T__inference_batch_normalization_2_layer_call_and_return_conditional_losses_107948377

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
!FusedBatchNormV3/ReadVariableOp_1Ê
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ0@:@:@:@:@:*
epsilon%o:*
is_training( 2
FusedBatchNormV3p
IdentityIdentityFusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ0@:::::W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@
 
_user_specified_nameinputs
¿
G
+__inference_re_lu_5_layer_call_fn_107948897

inputs
identityÒ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *O
fJRH
F__inference_re_lu_5_layer_call_and_return_conditional_losses_1079456962
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

×
'__inference_signature_wrapper_107947054

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
identity¢StatefulPartitionedCallÁ	
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
:ÿÿÿÿÿÿÿÿÿ`À*b
_read_only_resource_inputsD
B@	
 !"#$%&'()*+,-./0123456789:;<=>?@*2
config_proto" 

CPU

GPU2 *0J 8 *-
f(R&
$__inference__wrapped_model_1079437992
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`À2

Identity"
identityIdentity:output:0*±
_input_shapes
:ÿÿÿÿÿÿÿÿÿ`À::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:\ X
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`À
$
_user_specified_name
imageInput


R__inference_batch_normalization_layer_call_and_return_conditional_losses_107947999

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
!FusedBatchNormV3/ReadVariableOp_1Ü
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
is_training( 2
FusedBatchNormV3
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
®
­
E__inference_conv_5_layer_call_and_return_conditional_losses_107948580

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp¤
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
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
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddm
IdentityIdentityBiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:ÿÿÿÿÿÿÿÿÿ:::X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ô
b
F__inference_re_lu_2_layer_call_and_return_conditional_losses_107945337

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ0@:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@
 
_user_specified_nameinputs
Ô
b
F__inference_re_lu_6_layer_call_and_return_conditional_losses_107949062

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ0@:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@
 
_user_specified_nameinputs

¯
R__inference_batch_normalization_layer_call_and_return_conditional_losses_107945052

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity¢AssignNewValue¢AssignNewValue_1t
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
!FusedBatchNormV3/ReadVariableOp_1Ù
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*L
_output_shapes:
8:ÿÿÿÿÿÿÿÿÿ`À:::::*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3ÿ
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
:ÿÿÿÿÿÿÿÿÿ`À2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿ`À::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`À
 
_user_specified_nameinputs
Ó

T__inference_batch_normalization_7_layer_call_and_return_conditional_losses_107949201

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
!FusedBatchNormV3/ReadVariableOp_1Ê
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ0` : : : : :*
epsilon%o:*
is_training( 2
FusedBatchNormV3p
IdentityIdentityFusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0` 2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ0` :::::W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0` 
 
_user_specified_nameinputs
Ä

S__inference_tf_op_layer_concat_9_layer_call_and_return_conditional_losses_107949244
inputs_0
inputs_1
identityi
concat_9/axisConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
concat_9/axis
concat_9ConcatV2inputs_0inputs_1concat_9/axis:output:0*
N*
T0*
_cloned(*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`À 2

concat_9n
IdentityIdentityconcat_9:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`À 2

Identity"
identityIdentity:output:0*\
_input_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ`À:k g
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0:ZV
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`À
"
_user_specified_name
inputs/1
Ø
b
F__inference_re_lu_3_layer_call_and_return_conditional_losses_107948565

inputs
identityW
ReluReluinputs*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Reluo
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Æ
j
N__inference_tf_op_layer_Sub_layer_call_and_return_conditional_losses_107944999

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
:ÿÿÿÿÿÿÿÿÿ`À2
Subd
IdentityIdentitySub:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`À2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ`À:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`À
 
_user_specified_nameinputs


T__inference_batch_normalization_8_layer_call_and_return_conditional_losses_107944964

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
!FusedBatchNormV3/ReadVariableOp_1Ü
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
is_training( 2
FusedBatchNormV3
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
à
¬
9__inference_batch_normalization_1_layer_call_fn_107948233

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall©
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0` *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *]
fXRV
T__inference_batch_normalization_1_layer_call_and_return_conditional_losses_1079451652
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0` 2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ0` ::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0` 
 
_user_specified_nameinputs
¦
­
E__inference_conv_8_layer_call_and_return_conditional_losses_107945868

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
:ÿÿÿÿÿÿÿÿÿ0` *
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
:ÿÿÿÿÿÿÿÿÿ0` 2	
BiasAddl
IdentityIdentityBiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0` 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ0`@:::W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0`@
 
_user_specified_nameinputs
Ø

.__inference_upsample_6_layer_call_fn_107944415

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *R
fMRK
I__inference_upsample_6_layer_call_and_return_conditional_losses_1079444052
StatefulPartitionedCall©
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ::22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


*__inference_conv_4_layer_call_fn_107948432

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_conv_4_layer_call_and_return_conditional_losses_1079453562
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ@::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs


*__inference_conv_5_layer_call_fn_107948589

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_conv_5_layer_call_and_return_conditional_losses_1079454692
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:ÿÿÿÿÿÿÿÿÿ::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¯
Ü
0__inference_functional_1_layer_call_fn_107947787

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
identity¢StatefulPartitionedCallÒ	
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
:ÿÿÿÿÿÿÿÿÿ`À*P
_read_only_resource_inputs2
0.	
 !"#$'()*+,/01234789:;<?@*2
config_proto" 

CPU

GPU2 *0J 8 *T
fORM
K__inference_functional_1_layer_call_and_return_conditional_losses_1079464812
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`À2

Identity"
identityIdentity:output:0*±
_input_shapes
:ÿÿÿÿÿÿÿÿÿ`À::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`À
 
_user_specified_nameinputs

±
T__inference_batch_normalization_5_layer_call_and_return_conditional_losses_107945637

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity¢AssignNewValue¢AssignNewValue_1u
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
FusedBatchNormV3/ReadVariableOp®
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ý
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3ÿ
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
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

±
T__inference_batch_normalization_1_layer_call_and_return_conditional_losses_107948202

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity¢AssignNewValue¢AssignNewValue_1t
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
!FusedBatchNormV3/ReadVariableOp_1Ø
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ0` : : : : :*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3ÿ
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
:ÿÿÿÿÿÿÿÿÿ0` 2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ0` ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0` 
 
_user_specified_nameinputs
¨

T__inference_batch_normalization_5_layer_call_and_return_conditional_losses_107948797

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
FusedBatchNormV3/ReadVariableOp®
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1á
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
is_training( 2
FusedBatchNormV3
IdentityIdentityFusedBatchNormV3:y:0*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ä
¬
9__inference_batch_normalization_3_layer_call_fn_107948547

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCallª
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *]
fXRV
T__inference_batch_normalization_3_layer_call_and_return_conditional_losses_1079453912
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿ::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


T__inference_batch_normalization_2_layer_call_and_return_conditional_losses_107948313

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
!FusedBatchNormV3/ReadVariableOp_1Ü
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@:@:@:@:@:*
epsilon%o:*
is_training( 2
FusedBatchNormV3
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@:::::i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
¸
O
3__inference_max_pooling2d_2_layer_call_fn_107944147

inputs
identityô
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *W
fRRP
N__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_1079441412
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¬
¬
9__inference_batch_normalization_3_layer_call_fn_107948483

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall¼
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *]
fXRV
T__inference_batch_normalization_3_layer_call_and_return_conditional_losses_1079442092
StatefulPartitionedCall©
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ::::22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


*__inference_conv_1_layer_call_fn_107947961

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`À*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_conv_1_layer_call_and_return_conditional_losses_1079450172
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`À2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:ÿÿÿÿÿÿÿÿÿ`À::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`À
 
_user_specified_nameinputs
ç
n
R__inference_tf_op_layer_RealDiv_layer_call_and_return_conditional_losses_107947926

inputs
identity[
	RealDiv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ÿB2
	RealDiv/y
RealDivRealDivinputsRealDiv/y:output:0*
T0*
_cloned(*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`À2	
RealDivh
IdentityIdentityRealDiv:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`À2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ`À:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`À
 
_user_specified_nameinputs
Æ

S__inference_tf_op_layer_concat_6_layer_call_and_return_conditional_losses_107948734
inputs_0
inputs_1
identityi
concat_6/axisConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
concat_6/axis
concat_6ConcatV2inputs_0inputs_1concat_6/axis:output:0*
N*
T0*
_cloned(*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

concat_6n
IdentityIdentityconcat_6:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*]
_input_shapesL
J:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:l h
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0:ZV
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/1
±Ë
«
K__inference_functional_1_layer_call_and_return_conditional_losses_107946130

imageinput
conv_1_107945028
conv_1_107945030!
batch_normalization_107945097!
batch_normalization_107945099!
batch_normalization_107945101!
batch_normalization_107945103
conv_2_107945141
conv_2_107945143#
batch_normalization_1_107945210#
batch_normalization_1_107945212#
batch_normalization_1_107945214#
batch_normalization_1_107945216
conv_3_107945254
conv_3_107945256#
batch_normalization_2_107945323#
batch_normalization_2_107945325#
batch_normalization_2_107945327#
batch_normalization_2_107945329
conv_4_107945367
conv_4_107945369#
batch_normalization_3_107945436#
batch_normalization_3_107945438#
batch_normalization_3_107945440#
batch_normalization_3_107945442
conv_5_107945480
conv_5_107945482#
batch_normalization_4_107945549#
batch_normalization_4_107945551#
batch_normalization_4_107945553#
batch_normalization_4_107945555
upsample_6_107945571
upsample_6_107945573
conv_6_107945613
conv_6_107945615#
batch_normalization_5_107945682#
batch_normalization_5_107945684#
batch_normalization_5_107945686#
batch_normalization_5_107945688
upsample_7_107945704
upsample_7_107945706
conv_7_107945746
conv_7_107945748#
batch_normalization_6_107945815#
batch_normalization_6_107945817#
batch_normalization_6_107945819#
batch_normalization_6_107945821
upsample_8_107945837
upsample_8_107945839
conv_8_107945879
conv_8_107945881#
batch_normalization_7_107945948#
batch_normalization_7_107945950#
batch_normalization_7_107945952#
batch_normalization_7_107945954
upsample_9_107945970
upsample_9_107945972
conv_9_107946012
conv_9_107946014#
batch_normalization_8_107946081#
batch_normalization_8_107946083#
batch_normalization_8_107946085#
batch_normalization_8_107946087
final_107946124
final_107946126
identity¢+batch_normalization/StatefulPartitionedCall¢-batch_normalization_1/StatefulPartitionedCall¢-batch_normalization_2/StatefulPartitionedCall¢-batch_normalization_3/StatefulPartitionedCall¢-batch_normalization_4/StatefulPartitionedCall¢-batch_normalization_5/StatefulPartitionedCall¢-batch_normalization_6/StatefulPartitionedCall¢-batch_normalization_7/StatefulPartitionedCall¢-batch_normalization_8/StatefulPartitionedCall¢conv_1/StatefulPartitionedCall¢conv_2/StatefulPartitionedCall¢conv_3/StatefulPartitionedCall¢conv_4/StatefulPartitionedCall¢conv_5/StatefulPartitionedCall¢conv_6/StatefulPartitionedCall¢conv_7/StatefulPartitionedCall¢conv_8/StatefulPartitionedCall¢conv_9/StatefulPartitionedCall¢final/StatefulPartitionedCall¢"upsample_6/StatefulPartitionedCall¢"upsample_7/StatefulPartitionedCall¢"upsample_8/StatefulPartitionedCall¢"upsample_9/StatefulPartitionedCall
#tf_op_layer_RealDiv/PartitionedCallPartitionedCall
imageinput*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`À* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *[
fVRT
R__inference_tf_op_layer_RealDiv_layer_call_and_return_conditional_losses_1079449852%
#tf_op_layer_RealDiv/PartitionedCall 
tf_op_layer_Sub/PartitionedCallPartitionedCall,tf_op_layer_RealDiv/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`À* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *W
fRRP
N__inference_tf_op_layer_Sub_layer_call_and_return_conditional_losses_1079449992!
tf_op_layer_Sub/PartitionedCallÃ
conv_1/StatefulPartitionedCallStatefulPartitionedCall(tf_op_layer_Sub/PartitionedCall:output:0conv_1_107945028conv_1_107945030*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`À*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_conv_1_layer_call_and_return_conditional_losses_1079450172 
conv_1/StatefulPartitionedCallÃ
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall'conv_1/StatefulPartitionedCall:output:0batch_normalization_107945097batch_normalization_107945099batch_normalization_107945101batch_normalization_107945103*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`À*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *[
fVRT
R__inference_batch_normalization_layer_call_and_return_conditional_losses_1079450522-
+batch_normalization/StatefulPartitionedCall
re_lu/PartitionedCallPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`À* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *M
fHRF
D__inference_re_lu_layer_call_and_return_conditional_losses_1079451112
re_lu/PartitionedCall
max_pooling2d/PartitionedCallPartitionedCallre_lu/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0`* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *U
fPRN
L__inference_max_pooling2d_layer_call_and_return_conditional_losses_1079439092
max_pooling2d/PartitionedCallÀ
conv_2/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0conv_2_107945141conv_2_107945143*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0` *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_conv_2_layer_call_and_return_conditional_losses_1079451302 
conv_2/StatefulPartitionedCallÐ
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall'conv_2/StatefulPartitionedCall:output:0batch_normalization_1_107945210batch_normalization_1_107945212batch_normalization_1_107945214batch_normalization_1_107945216*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0` *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *]
fXRV
T__inference_batch_normalization_1_layer_call_and_return_conditional_losses_1079451652/
-batch_normalization_1/StatefulPartitionedCall
re_lu_1/PartitionedCallPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0` * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *O
fJRH
F__inference_re_lu_1_layer_call_and_return_conditional_losses_1079452242
re_lu_1/PartitionedCall
max_pooling2d_1/PartitionedCallPartitionedCall re_lu_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0 * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *W
fRRP
N__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_1079440252!
max_pooling2d_1/PartitionedCallÂ
conv_3/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_1/PartitionedCall:output:0conv_3_107945254conv_3_107945256*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_conv_3_layer_call_and_return_conditional_losses_1079452432 
conv_3/StatefulPartitionedCallÐ
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall'conv_3/StatefulPartitionedCall:output:0batch_normalization_2_107945323batch_normalization_2_107945325batch_normalization_2_107945327batch_normalization_2_107945329*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *]
fXRV
T__inference_batch_normalization_2_layer_call_and_return_conditional_losses_1079452782/
-batch_normalization_2/StatefulPartitionedCall
re_lu_2/PartitionedCallPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *O
fJRH
F__inference_re_lu_2_layer_call_and_return_conditional_losses_1079453372
re_lu_2/PartitionedCall
max_pooling2d_2/PartitionedCallPartitionedCall re_lu_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *W
fRRP
N__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_1079441412!
max_pooling2d_2/PartitionedCallÃ
conv_4/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_2/PartitionedCall:output:0conv_4_107945367conv_4_107945369*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_conv_4_layer_call_and_return_conditional_losses_1079453562 
conv_4/StatefulPartitionedCallÑ
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCall'conv_4/StatefulPartitionedCall:output:0batch_normalization_3_107945436batch_normalization_3_107945438batch_normalization_3_107945440batch_normalization_3_107945442*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *]
fXRV
T__inference_batch_normalization_3_layer_call_and_return_conditional_losses_1079453912/
-batch_normalization_3/StatefulPartitionedCall
re_lu_3/PartitionedCallPartitionedCall6batch_normalization_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *O
fJRH
F__inference_re_lu_3_layer_call_and_return_conditional_losses_1079454502
re_lu_3/PartitionedCall
max_pooling2d_3/PartitionedCallPartitionedCall re_lu_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *W
fRRP
N__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_1079442572!
max_pooling2d_3/PartitionedCallÃ
conv_5/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_3/PartitionedCall:output:0conv_5_107945480conv_5_107945482*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_conv_5_layer_call_and_return_conditional_losses_1079454692 
conv_5/StatefulPartitionedCallÑ
-batch_normalization_4/StatefulPartitionedCallStatefulPartitionedCall'conv_5/StatefulPartitionedCall:output:0batch_normalization_4_107945549batch_normalization_4_107945551batch_normalization_4_107945553batch_normalization_4_107945555*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *]
fXRV
T__inference_batch_normalization_4_layer_call_and_return_conditional_losses_1079455042/
-batch_normalization_4/StatefulPartitionedCall
re_lu_4/PartitionedCallPartitionedCall6batch_normalization_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *O
fJRH
F__inference_re_lu_4_layer_call_and_return_conditional_losses_1079455632
re_lu_4/PartitionedCallá
"upsample_6/StatefulPartitionedCallStatefulPartitionedCall re_lu_4/PartitionedCall:output:0upsample_6_107945571upsample_6_107945573*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *R
fMRK
I__inference_upsample_6_layer_call_and_return_conditional_losses_1079444052$
"upsample_6/StatefulPartitionedCallÑ
$tf_op_layer_concat_6/PartitionedCallPartitionedCall+upsample_6/StatefulPartitionedCall:output:0 re_lu_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *\
fWRU
S__inference_tf_op_layer_concat_6_layer_call_and_return_conditional_losses_1079455832&
$tf_op_layer_concat_6/PartitionedCallÈ
conv_6/StatefulPartitionedCallStatefulPartitionedCall-tf_op_layer_concat_6/PartitionedCall:output:0conv_6_107945613conv_6_107945615*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_conv_6_layer_call_and_return_conditional_losses_1079456022 
conv_6/StatefulPartitionedCallÑ
-batch_normalization_5/StatefulPartitionedCallStatefulPartitionedCall'conv_6/StatefulPartitionedCall:output:0batch_normalization_5_107945682batch_normalization_5_107945684batch_normalization_5_107945686batch_normalization_5_107945688*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *]
fXRV
T__inference_batch_normalization_5_layer_call_and_return_conditional_losses_1079456372/
-batch_normalization_5/StatefulPartitionedCall
re_lu_5/PartitionedCallPartitionedCall6batch_normalization_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *O
fJRH
F__inference_re_lu_5_layer_call_and_return_conditional_losses_1079456962
re_lu_5/PartitionedCallà
"upsample_7/StatefulPartitionedCallStatefulPartitionedCall re_lu_5/PartitionedCall:output:0upsample_7_107945704upsample_7_107945706*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *R
fMRK
I__inference_upsample_7_layer_call_and_return_conditional_losses_1079445572$
"upsample_7/StatefulPartitionedCallÑ
$tf_op_layer_concat_7/PartitionedCallPartitionedCall+upsample_7/StatefulPartitionedCall:output:0 re_lu_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *\
fWRU
S__inference_tf_op_layer_concat_7_layer_call_and_return_conditional_losses_1079457162&
$tf_op_layer_concat_7/PartitionedCallÇ
conv_7/StatefulPartitionedCallStatefulPartitionedCall-tf_op_layer_concat_7/PartitionedCall:output:0conv_7_107945746conv_7_107945748*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_conv_7_layer_call_and_return_conditional_losses_1079457352 
conv_7/StatefulPartitionedCallÐ
-batch_normalization_6/StatefulPartitionedCallStatefulPartitionedCall'conv_7/StatefulPartitionedCall:output:0batch_normalization_6_107945815batch_normalization_6_107945817batch_normalization_6_107945819batch_normalization_6_107945821*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *]
fXRV
T__inference_batch_normalization_6_layer_call_and_return_conditional_losses_1079457702/
-batch_normalization_6/StatefulPartitionedCall
re_lu_6/PartitionedCallPartitionedCall6batch_normalization_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *O
fJRH
F__inference_re_lu_6_layer_call_and_return_conditional_losses_1079458292
re_lu_6/PartitionedCallà
"upsample_8/StatefulPartitionedCallStatefulPartitionedCall re_lu_6/PartitionedCall:output:0upsample_8_107945837upsample_8_107945839*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *R
fMRK
I__inference_upsample_8_layer_call_and_return_conditional_losses_1079447092$
"upsample_8/StatefulPartitionedCallÐ
$tf_op_layer_concat_8/PartitionedCallPartitionedCall+upsample_8/StatefulPartitionedCall:output:0 re_lu_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0`@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *\
fWRU
S__inference_tf_op_layer_concat_8_layer_call_and_return_conditional_losses_1079458492&
$tf_op_layer_concat_8/PartitionedCallÇ
conv_8/StatefulPartitionedCallStatefulPartitionedCall-tf_op_layer_concat_8/PartitionedCall:output:0conv_8_107945879conv_8_107945881*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0` *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_conv_8_layer_call_and_return_conditional_losses_1079458682 
conv_8/StatefulPartitionedCallÐ
-batch_normalization_7/StatefulPartitionedCallStatefulPartitionedCall'conv_8/StatefulPartitionedCall:output:0batch_normalization_7_107945948batch_normalization_7_107945950batch_normalization_7_107945952batch_normalization_7_107945954*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0` *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *]
fXRV
T__inference_batch_normalization_7_layer_call_and_return_conditional_losses_1079459032/
-batch_normalization_7/StatefulPartitionedCall
re_lu_7/PartitionedCallPartitionedCall6batch_normalization_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0` * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *O
fJRH
F__inference_re_lu_7_layer_call_and_return_conditional_losses_1079459622
re_lu_7/PartitionedCallà
"upsample_9/StatefulPartitionedCallStatefulPartitionedCall re_lu_7/PartitionedCall:output:0upsample_9_107945970upsample_9_107945972*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *R
fMRK
I__inference_upsample_9_layer_call_and_return_conditional_losses_1079448612$
"upsample_9/StatefulPartitionedCallÏ
$tf_op_layer_concat_9/PartitionedCallPartitionedCall+upsample_9/StatefulPartitionedCall:output:0re_lu/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`À * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *\
fWRU
S__inference_tf_op_layer_concat_9_layer_call_and_return_conditional_losses_1079459822&
$tf_op_layer_concat_9/PartitionedCallÈ
conv_9/StatefulPartitionedCallStatefulPartitionedCall-tf_op_layer_concat_9/PartitionedCall:output:0conv_9_107946012conv_9_107946014*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`À*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_conv_9_layer_call_and_return_conditional_losses_1079460012 
conv_9/StatefulPartitionedCallÑ
-batch_normalization_8/StatefulPartitionedCallStatefulPartitionedCall'conv_9/StatefulPartitionedCall:output:0batch_normalization_8_107946081batch_normalization_8_107946083batch_normalization_8_107946085batch_normalization_8_107946087*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`À*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *]
fXRV
T__inference_batch_normalization_8_layer_call_and_return_conditional_losses_1079460362/
-batch_normalization_8/StatefulPartitionedCall
re_lu_8/PartitionedCallPartitionedCall6batch_normalization_8/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`À* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *O
fJRH
F__inference_re_lu_8_layer_call_and_return_conditional_losses_1079460952
re_lu_8/PartitionedCall¶
final/StatefulPartitionedCallStatefulPartitionedCall re_lu_8/PartitionedCall:output:0final_107946124final_107946126*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`À*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *M
fHRF
D__inference_final_layer_call_and_return_conditional_losses_1079461132
final/StatefulPartitionedCall
IdentityIdentity&final/StatefulPartitionedCall:output:0,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall.^batch_normalization_3/StatefulPartitionedCall.^batch_normalization_4/StatefulPartitionedCall.^batch_normalization_5/StatefulPartitionedCall.^batch_normalization_6/StatefulPartitionedCall.^batch_normalization_7/StatefulPartitionedCall.^batch_normalization_8/StatefulPartitionedCall^conv_1/StatefulPartitionedCall^conv_2/StatefulPartitionedCall^conv_3/StatefulPartitionedCall^conv_4/StatefulPartitionedCall^conv_5/StatefulPartitionedCall^conv_6/StatefulPartitionedCall^conv_7/StatefulPartitionedCall^conv_8/StatefulPartitionedCall^conv_9/StatefulPartitionedCall^final/StatefulPartitionedCall#^upsample_6/StatefulPartitionedCall#^upsample_7/StatefulPartitionedCall#^upsample_8/StatefulPartitionedCall#^upsample_9/StatefulPartitionedCall*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`À2

Identity"
identityIdentity:output:0*±
_input_shapes
:ÿÿÿÿÿÿÿÿÿ`À::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::2Z
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
:ÿÿÿÿÿÿÿÿÿ`À
$
_user_specified_name
imageInput
«
­
E__inference_conv_4_layer_call_and_return_conditional_losses_107945356

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@*
dtype02
Conv2D/ReadVariableOp¤
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
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
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddm
IdentityIdentityBiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ@:::W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
ª
¬
9__inference_batch_normalization_6_layer_call_fn_107948993

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall½
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *]
fXRV
T__inference_batch_normalization_6_layer_call_and_return_conditional_losses_1079446602
StatefulPartitionedCall¨
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
Ø
±
T__inference_batch_normalization_4_layer_call_and_return_conditional_losses_107948609

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity¢AssignNewValue¢AssignNewValue_1u
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
FusedBatchNormV3/ReadVariableOp®
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ï
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3ÿ
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
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Â

S__inference_tf_op_layer_concat_7_layer_call_and_return_conditional_losses_107948904
inputs_0
inputs_1
identityi
concat_7/axisConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
concat_7/axis
concat_7ConcatV2inputs_0inputs_1concat_7/axis:output:0*
N*
T0*
_cloned(*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ02

concat_7n
IdentityIdentityconcat_7:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ02

Identity"
identityIdentity:output:0*[
_input_shapesJ
H:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@:ÿÿÿÿÿÿÿÿÿ0@:k g
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@
"
_user_specified_name
inputs/1
ß

T__inference_batch_normalization_4_layer_call_and_return_conditional_losses_107945522

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
FusedBatchNormV3/ReadVariableOp®
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ï
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
is_training( 2
FusedBatchNormV3q
IdentityIdentityFusedBatchNormV3:y:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿ:::::X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

±
T__inference_batch_normalization_8_layer_call_and_return_conditional_losses_107949353

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity¢AssignNewValue¢AssignNewValue_1t
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
!FusedBatchNormV3/ReadVariableOp_1Ù
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*L
_output_shapes:
8:ÿÿÿÿÿÿÿÿÿ`À:::::*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3ÿ
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
:ÿÿÿÿÿÿÿÿÿ`À2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿ`À::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`À
 
_user_specified_nameinputs
ç
n
R__inference_tf_op_layer_RealDiv_layer_call_and_return_conditional_losses_107944985

inputs
identity[
	RealDiv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ÿB2
	RealDiv/y
RealDivRealDivinputsRealDiv/y:output:0*
T0*
_cloned(*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`À2	
RealDivh
IdentityIdentityRealDiv:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`À2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ`À:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`À
 
_user_specified_nameinputs

±
T__inference_batch_normalization_3_layer_call_and_return_conditional_losses_107948516

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity¢AssignNewValue¢AssignNewValue_1u
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
FusedBatchNormV3/ReadVariableOp®
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ý
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3ÿ
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
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
×

T__inference_batch_normalization_8_layer_call_and_return_conditional_losses_107946054

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
!FusedBatchNormV3/ReadVariableOp_1Ë
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*L
_output_shapes:
8:ÿÿÿÿÿÿÿÿÿ`À:::::*
epsilon%o:*
is_training( 2
FusedBatchNormV3q
IdentityIdentityFusedBatchNormV3:y:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`À2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿ`À:::::X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`À
 
_user_specified_nameinputs
»
G
+__inference_re_lu_2_layer_call_fn_107948413

inputs
identityÑ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *O
fJRH
F__inference_re_lu_2_layer_call_and_return_conditional_losses_1079453372
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ0@:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@
 
_user_specified_nameinputs
Ø
±
T__inference_batch_normalization_3_layer_call_and_return_conditional_losses_107948452

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity¢AssignNewValue¢AssignNewValue_1u
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
FusedBatchNormV3/ReadVariableOp®
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ï
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3ÿ
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
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ì
±
T__inference_batch_normalization_7_layer_call_and_return_conditional_losses_107944781

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity¢AssignNewValue¢AssignNewValue_1t
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
!FusedBatchNormV3/ReadVariableOp_1ê
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ : : : : :*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3ÿ
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
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs

h
L__inference_max_pooling2d_layer_call_and_return_conditional_losses_107943909

inputs
identity­
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
2	
MaxPool
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¦
­
E__inference_conv_3_layer_call_and_return_conditional_losses_107945243

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
:ÿÿÿÿÿÿÿÿÿ0@*
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
:ÿÿÿÿÿÿÿÿÿ0@2	
BiasAddl
IdentityIdentityBiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ0 :::W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0 
 
_user_specified_nameinputs
Ó

T__inference_batch_normalization_6_layer_call_and_return_conditional_losses_107949031

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
!FusedBatchNormV3/ReadVariableOp_1Ê
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ0@:@:@:@:@:*
epsilon%o:*
is_training( 2
FusedBatchNormV3p
IdentityIdentityFusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ0@:::::W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@
 
_user_specified_nameinputs
Ø
b
F__inference_re_lu_3_layer_call_and_return_conditional_losses_107945450

inputs
identityW
ReluReluinputs*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Reluo
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
®
­
E__inference_conv_6_layer_call_and_return_conditional_losses_107945602

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp¤
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
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
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddm
IdentityIdentityBiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:ÿÿÿÿÿÿÿÿÿ:::X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ì
±
T__inference_batch_normalization_8_layer_call_and_return_conditional_losses_107944933

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity¢AssignNewValue¢AssignNewValue_1t
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
!FusedBatchNormV3/ReadVariableOp_1ê
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3ÿ
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
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¨
¬
9__inference_batch_normalization_2_layer_call_fn_107948326

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall»
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *]
fXRV
T__inference_batch_normalization_2_layer_call_and_return_conditional_losses_1079440932
StatefulPartitionedCall¨
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
É$
»
I__inference_upsample_9_layer_call_and_return_conditional_losses_107944861

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
strided_slice/stack_2â
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
strided_slice_1/stack_2ì
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
strided_slice_2/stack_2ì
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
strided_slice_3/stack_2ì
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
conv2d_transpose/ReadVariableOpñ
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides
2
conv2d_transpose
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp¤
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2	
BiasAdd~
IdentityIdentityBiasAdd:output:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ :::i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
Ô
b
F__inference_re_lu_7_layer_call_and_return_conditional_losses_107949232

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0` 2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0` 2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ0` :W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0` 
 
_user_specified_nameinputs
Ø
b
F__inference_re_lu_5_layer_call_and_return_conditional_losses_107948892

inputs
identityW
ReluReluinputs*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Reluo
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

j
N__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_107944025

inputs
identity­
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
2	
MaxPool
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

±
T__inference_batch_normalization_4_layer_call_and_return_conditional_losses_107945504

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity¢AssignNewValue¢AssignNewValue_1u
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
FusedBatchNormV3/ReadVariableOp®
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ý
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3ÿ
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
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
«
­
E__inference_conv_4_layer_call_and_return_conditional_losses_107948423

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@*
dtype02
Conv2D/ReadVariableOp¤
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
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
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddm
IdentityIdentityBiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ@:::W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
·Ë
§
K__inference_functional_1_layer_call_and_return_conditional_losses_107946788

inputs
conv_1_107946619
conv_1_107946621!
batch_normalization_107946624!
batch_normalization_107946626!
batch_normalization_107946628!
batch_normalization_107946630
conv_2_107946635
conv_2_107946637#
batch_normalization_1_107946640#
batch_normalization_1_107946642#
batch_normalization_1_107946644#
batch_normalization_1_107946646
conv_3_107946651
conv_3_107946653#
batch_normalization_2_107946656#
batch_normalization_2_107946658#
batch_normalization_2_107946660#
batch_normalization_2_107946662
conv_4_107946667
conv_4_107946669#
batch_normalization_3_107946672#
batch_normalization_3_107946674#
batch_normalization_3_107946676#
batch_normalization_3_107946678
conv_5_107946683
conv_5_107946685#
batch_normalization_4_107946688#
batch_normalization_4_107946690#
batch_normalization_4_107946692#
batch_normalization_4_107946694
upsample_6_107946698
upsample_6_107946700
conv_6_107946704
conv_6_107946706#
batch_normalization_5_107946709#
batch_normalization_5_107946711#
batch_normalization_5_107946713#
batch_normalization_5_107946715
upsample_7_107946719
upsample_7_107946721
conv_7_107946725
conv_7_107946727#
batch_normalization_6_107946730#
batch_normalization_6_107946732#
batch_normalization_6_107946734#
batch_normalization_6_107946736
upsample_8_107946740
upsample_8_107946742
conv_8_107946746
conv_8_107946748#
batch_normalization_7_107946751#
batch_normalization_7_107946753#
batch_normalization_7_107946755#
batch_normalization_7_107946757
upsample_9_107946761
upsample_9_107946763
conv_9_107946767
conv_9_107946769#
batch_normalization_8_107946772#
batch_normalization_8_107946774#
batch_normalization_8_107946776#
batch_normalization_8_107946778
final_107946782
final_107946784
identity¢+batch_normalization/StatefulPartitionedCall¢-batch_normalization_1/StatefulPartitionedCall¢-batch_normalization_2/StatefulPartitionedCall¢-batch_normalization_3/StatefulPartitionedCall¢-batch_normalization_4/StatefulPartitionedCall¢-batch_normalization_5/StatefulPartitionedCall¢-batch_normalization_6/StatefulPartitionedCall¢-batch_normalization_7/StatefulPartitionedCall¢-batch_normalization_8/StatefulPartitionedCall¢conv_1/StatefulPartitionedCall¢conv_2/StatefulPartitionedCall¢conv_3/StatefulPartitionedCall¢conv_4/StatefulPartitionedCall¢conv_5/StatefulPartitionedCall¢conv_6/StatefulPartitionedCall¢conv_7/StatefulPartitionedCall¢conv_8/StatefulPartitionedCall¢conv_9/StatefulPartitionedCall¢final/StatefulPartitionedCall¢"upsample_6/StatefulPartitionedCall¢"upsample_7/StatefulPartitionedCall¢"upsample_8/StatefulPartitionedCall¢"upsample_9/StatefulPartitionedCall
#tf_op_layer_RealDiv/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`À* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *[
fVRT
R__inference_tf_op_layer_RealDiv_layer_call_and_return_conditional_losses_1079449852%
#tf_op_layer_RealDiv/PartitionedCall 
tf_op_layer_Sub/PartitionedCallPartitionedCall,tf_op_layer_RealDiv/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`À* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *W
fRRP
N__inference_tf_op_layer_Sub_layer_call_and_return_conditional_losses_1079449992!
tf_op_layer_Sub/PartitionedCallÃ
conv_1/StatefulPartitionedCallStatefulPartitionedCall(tf_op_layer_Sub/PartitionedCall:output:0conv_1_107946619conv_1_107946621*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`À*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_conv_1_layer_call_and_return_conditional_losses_1079450172 
conv_1/StatefulPartitionedCallÅ
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall'conv_1/StatefulPartitionedCall:output:0batch_normalization_107946624batch_normalization_107946626batch_normalization_107946628batch_normalization_107946630*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`À*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *[
fVRT
R__inference_batch_normalization_layer_call_and_return_conditional_losses_1079450702-
+batch_normalization/StatefulPartitionedCall
re_lu/PartitionedCallPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`À* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *M
fHRF
D__inference_re_lu_layer_call_and_return_conditional_losses_1079451112
re_lu/PartitionedCall
max_pooling2d/PartitionedCallPartitionedCallre_lu/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0`* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *U
fPRN
L__inference_max_pooling2d_layer_call_and_return_conditional_losses_1079439092
max_pooling2d/PartitionedCallÀ
conv_2/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0conv_2_107946635conv_2_107946637*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0` *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_conv_2_layer_call_and_return_conditional_losses_1079451302 
conv_2/StatefulPartitionedCallÒ
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall'conv_2/StatefulPartitionedCall:output:0batch_normalization_1_107946640batch_normalization_1_107946642batch_normalization_1_107946644batch_normalization_1_107946646*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0` *&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *]
fXRV
T__inference_batch_normalization_1_layer_call_and_return_conditional_losses_1079451832/
-batch_normalization_1/StatefulPartitionedCall
re_lu_1/PartitionedCallPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0` * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *O
fJRH
F__inference_re_lu_1_layer_call_and_return_conditional_losses_1079452242
re_lu_1/PartitionedCall
max_pooling2d_1/PartitionedCallPartitionedCall re_lu_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0 * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *W
fRRP
N__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_1079440252!
max_pooling2d_1/PartitionedCallÂ
conv_3/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_1/PartitionedCall:output:0conv_3_107946651conv_3_107946653*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_conv_3_layer_call_and_return_conditional_losses_1079452432 
conv_3/StatefulPartitionedCallÒ
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall'conv_3/StatefulPartitionedCall:output:0batch_normalization_2_107946656batch_normalization_2_107946658batch_normalization_2_107946660batch_normalization_2_107946662*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *]
fXRV
T__inference_batch_normalization_2_layer_call_and_return_conditional_losses_1079452962/
-batch_normalization_2/StatefulPartitionedCall
re_lu_2/PartitionedCallPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *O
fJRH
F__inference_re_lu_2_layer_call_and_return_conditional_losses_1079453372
re_lu_2/PartitionedCall
max_pooling2d_2/PartitionedCallPartitionedCall re_lu_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *W
fRRP
N__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_1079441412!
max_pooling2d_2/PartitionedCallÃ
conv_4/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_2/PartitionedCall:output:0conv_4_107946667conv_4_107946669*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_conv_4_layer_call_and_return_conditional_losses_1079453562 
conv_4/StatefulPartitionedCallÓ
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCall'conv_4/StatefulPartitionedCall:output:0batch_normalization_3_107946672batch_normalization_3_107946674batch_normalization_3_107946676batch_normalization_3_107946678*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *]
fXRV
T__inference_batch_normalization_3_layer_call_and_return_conditional_losses_1079454092/
-batch_normalization_3/StatefulPartitionedCall
re_lu_3/PartitionedCallPartitionedCall6batch_normalization_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *O
fJRH
F__inference_re_lu_3_layer_call_and_return_conditional_losses_1079454502
re_lu_3/PartitionedCall
max_pooling2d_3/PartitionedCallPartitionedCall re_lu_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *W
fRRP
N__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_1079442572!
max_pooling2d_3/PartitionedCallÃ
conv_5/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_3/PartitionedCall:output:0conv_5_107946683conv_5_107946685*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_conv_5_layer_call_and_return_conditional_losses_1079454692 
conv_5/StatefulPartitionedCallÓ
-batch_normalization_4/StatefulPartitionedCallStatefulPartitionedCall'conv_5/StatefulPartitionedCall:output:0batch_normalization_4_107946688batch_normalization_4_107946690batch_normalization_4_107946692batch_normalization_4_107946694*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *]
fXRV
T__inference_batch_normalization_4_layer_call_and_return_conditional_losses_1079455222/
-batch_normalization_4/StatefulPartitionedCall
re_lu_4/PartitionedCallPartitionedCall6batch_normalization_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *O
fJRH
F__inference_re_lu_4_layer_call_and_return_conditional_losses_1079455632
re_lu_4/PartitionedCallá
"upsample_6/StatefulPartitionedCallStatefulPartitionedCall re_lu_4/PartitionedCall:output:0upsample_6_107946698upsample_6_107946700*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *R
fMRK
I__inference_upsample_6_layer_call_and_return_conditional_losses_1079444052$
"upsample_6/StatefulPartitionedCallÑ
$tf_op_layer_concat_6/PartitionedCallPartitionedCall+upsample_6/StatefulPartitionedCall:output:0 re_lu_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *\
fWRU
S__inference_tf_op_layer_concat_6_layer_call_and_return_conditional_losses_1079455832&
$tf_op_layer_concat_6/PartitionedCallÈ
conv_6/StatefulPartitionedCallStatefulPartitionedCall-tf_op_layer_concat_6/PartitionedCall:output:0conv_6_107946704conv_6_107946706*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_conv_6_layer_call_and_return_conditional_losses_1079456022 
conv_6/StatefulPartitionedCallÓ
-batch_normalization_5/StatefulPartitionedCallStatefulPartitionedCall'conv_6/StatefulPartitionedCall:output:0batch_normalization_5_107946709batch_normalization_5_107946711batch_normalization_5_107946713batch_normalization_5_107946715*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *]
fXRV
T__inference_batch_normalization_5_layer_call_and_return_conditional_losses_1079456552/
-batch_normalization_5/StatefulPartitionedCall
re_lu_5/PartitionedCallPartitionedCall6batch_normalization_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *O
fJRH
F__inference_re_lu_5_layer_call_and_return_conditional_losses_1079456962
re_lu_5/PartitionedCallà
"upsample_7/StatefulPartitionedCallStatefulPartitionedCall re_lu_5/PartitionedCall:output:0upsample_7_107946719upsample_7_107946721*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *R
fMRK
I__inference_upsample_7_layer_call_and_return_conditional_losses_1079445572$
"upsample_7/StatefulPartitionedCallÑ
$tf_op_layer_concat_7/PartitionedCallPartitionedCall+upsample_7/StatefulPartitionedCall:output:0 re_lu_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *\
fWRU
S__inference_tf_op_layer_concat_7_layer_call_and_return_conditional_losses_1079457162&
$tf_op_layer_concat_7/PartitionedCallÇ
conv_7/StatefulPartitionedCallStatefulPartitionedCall-tf_op_layer_concat_7/PartitionedCall:output:0conv_7_107946725conv_7_107946727*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_conv_7_layer_call_and_return_conditional_losses_1079457352 
conv_7/StatefulPartitionedCallÒ
-batch_normalization_6/StatefulPartitionedCallStatefulPartitionedCall'conv_7/StatefulPartitionedCall:output:0batch_normalization_6_107946730batch_normalization_6_107946732batch_normalization_6_107946734batch_normalization_6_107946736*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *]
fXRV
T__inference_batch_normalization_6_layer_call_and_return_conditional_losses_1079457882/
-batch_normalization_6/StatefulPartitionedCall
re_lu_6/PartitionedCallPartitionedCall6batch_normalization_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *O
fJRH
F__inference_re_lu_6_layer_call_and_return_conditional_losses_1079458292
re_lu_6/PartitionedCallà
"upsample_8/StatefulPartitionedCallStatefulPartitionedCall re_lu_6/PartitionedCall:output:0upsample_8_107946740upsample_8_107946742*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *R
fMRK
I__inference_upsample_8_layer_call_and_return_conditional_losses_1079447092$
"upsample_8/StatefulPartitionedCallÐ
$tf_op_layer_concat_8/PartitionedCallPartitionedCall+upsample_8/StatefulPartitionedCall:output:0 re_lu_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0`@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *\
fWRU
S__inference_tf_op_layer_concat_8_layer_call_and_return_conditional_losses_1079458492&
$tf_op_layer_concat_8/PartitionedCallÇ
conv_8/StatefulPartitionedCallStatefulPartitionedCall-tf_op_layer_concat_8/PartitionedCall:output:0conv_8_107946746conv_8_107946748*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0` *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_conv_8_layer_call_and_return_conditional_losses_1079458682 
conv_8/StatefulPartitionedCallÒ
-batch_normalization_7/StatefulPartitionedCallStatefulPartitionedCall'conv_8/StatefulPartitionedCall:output:0batch_normalization_7_107946751batch_normalization_7_107946753batch_normalization_7_107946755batch_normalization_7_107946757*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0` *&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *]
fXRV
T__inference_batch_normalization_7_layer_call_and_return_conditional_losses_1079459212/
-batch_normalization_7/StatefulPartitionedCall
re_lu_7/PartitionedCallPartitionedCall6batch_normalization_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0` * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *O
fJRH
F__inference_re_lu_7_layer_call_and_return_conditional_losses_1079459622
re_lu_7/PartitionedCallà
"upsample_9/StatefulPartitionedCallStatefulPartitionedCall re_lu_7/PartitionedCall:output:0upsample_9_107946761upsample_9_107946763*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *R
fMRK
I__inference_upsample_9_layer_call_and_return_conditional_losses_1079448612$
"upsample_9/StatefulPartitionedCallÏ
$tf_op_layer_concat_9/PartitionedCallPartitionedCall+upsample_9/StatefulPartitionedCall:output:0re_lu/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`À * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *\
fWRU
S__inference_tf_op_layer_concat_9_layer_call_and_return_conditional_losses_1079459822&
$tf_op_layer_concat_9/PartitionedCallÈ
conv_9/StatefulPartitionedCallStatefulPartitionedCall-tf_op_layer_concat_9/PartitionedCall:output:0conv_9_107946767conv_9_107946769*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`À*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_conv_9_layer_call_and_return_conditional_losses_1079460012 
conv_9/StatefulPartitionedCallÓ
-batch_normalization_8/StatefulPartitionedCallStatefulPartitionedCall'conv_9/StatefulPartitionedCall:output:0batch_normalization_8_107946772batch_normalization_8_107946774batch_normalization_8_107946776batch_normalization_8_107946778*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`À*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *]
fXRV
T__inference_batch_normalization_8_layer_call_and_return_conditional_losses_1079460542/
-batch_normalization_8/StatefulPartitionedCall
re_lu_8/PartitionedCallPartitionedCall6batch_normalization_8/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`À* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *O
fJRH
F__inference_re_lu_8_layer_call_and_return_conditional_losses_1079460952
re_lu_8/PartitionedCall¶
final/StatefulPartitionedCallStatefulPartitionedCall re_lu_8/PartitionedCall:output:0final_107946782final_107946784*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`À*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *M
fHRF
D__inference_final_layer_call_and_return_conditional_losses_1079461132
final/StatefulPartitionedCall
IdentityIdentity&final/StatefulPartitionedCall:output:0,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall.^batch_normalization_3/StatefulPartitionedCall.^batch_normalization_4/StatefulPartitionedCall.^batch_normalization_5/StatefulPartitionedCall.^batch_normalization_6/StatefulPartitionedCall.^batch_normalization_7/StatefulPartitionedCall.^batch_normalization_8/StatefulPartitionedCall^conv_1/StatefulPartitionedCall^conv_2/StatefulPartitionedCall^conv_3/StatefulPartitionedCall^conv_4/StatefulPartitionedCall^conv_5/StatefulPartitionedCall^conv_6/StatefulPartitionedCall^conv_7/StatefulPartitionedCall^conv_8/StatefulPartitionedCall^conv_9/StatefulPartitionedCall^final/StatefulPartitionedCall#^upsample_6/StatefulPartitionedCall#^upsample_7/StatefulPartitionedCall#^upsample_8/StatefulPartitionedCall#^upsample_9/StatefulPartitionedCall*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`À2

Identity"
identityIdentity:output:0*±
_input_shapes
:ÿÿÿÿÿÿÿÿÿ`À::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::2Z
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
:ÿÿÿÿÿÿÿÿÿ`À
 
_user_specified_nameinputs
ª
¬
D__inference_final_layer_call_and_return_conditional_losses_107949417

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp¤
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`À*
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
:ÿÿÿÿÿÿÿÿÿ`À2	
BiasAddm
IdentityIdentityBiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`À2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:ÿÿÿÿÿÿÿÿÿ`À:::X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`À
 
_user_specified_nameinputs
»
G
+__inference_re_lu_7_layer_call_fn_107949237

inputs
identityÑ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0` * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *O
fJRH
F__inference_re_lu_7_layer_call_and_return_conditional_losses_1079459622
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0` 2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ0` :W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0` 
 
_user_specified_nameinputs
Ô
b
F__inference_re_lu_1_layer_call_and_return_conditional_losses_107945224

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0` 2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0` 2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ0` :W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0` 
 
_user_specified_nameinputs
Ì
±
T__inference_batch_normalization_1_layer_call_and_return_conditional_losses_107948138

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity¢AssignNewValue¢AssignNewValue_1t
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
!FusedBatchNormV3/ReadVariableOp_1ê
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ : : : : :*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3ÿ
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
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
¨

T__inference_batch_normalization_5_layer_call_and_return_conditional_losses_107944508

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
FusedBatchNormV3/ReadVariableOp®
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1á
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
is_training( 2
FusedBatchNormV3
IdentityIdentityFusedBatchNormV3:y:0*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
æ
¬
9__inference_batch_normalization_3_layer_call_fn_107948560

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall¬
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *]
fXRV
T__inference_batch_normalization_3_layer_call_and_return_conditional_losses_1079454092
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿ::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¨
¬
9__inference_batch_normalization_6_layer_call_fn_107948980

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall»
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *]
fXRV
T__inference_batch_normalization_6_layer_call_and_return_conditional_losses_1079446292
StatefulPartitionedCall¨
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
»
G
+__inference_re_lu_6_layer_call_fn_107949067

inputs
identityÑ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *O
fJRH
F__inference_re_lu_6_layer_call_and_return_conditional_losses_1079458292
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ0@:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@
 
_user_specified_nameinputs
¿
G
+__inference_re_lu_4_layer_call_fn_107948727

inputs
identityÒ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *O
fJRH
F__inference_re_lu_4_layer_call_and_return_conditional_losses_1079455632
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

±
T__inference_batch_normalization_7_layer_call_and_return_conditional_losses_107945903

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity¢AssignNewValue¢AssignNewValue_1t
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
!FusedBatchNormV3/ReadVariableOp_1Ø
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ0` : : : : :*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3ÿ
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
:ÿÿÿÿÿÿÿÿÿ0` 2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ0` ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0` 
 
_user_specified_nameinputs
â
¬
9__inference_batch_normalization_7_layer_call_fn_107949227

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall«
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0` *&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *]
fXRV
T__inference_batch_normalization_7_layer_call_and_return_conditional_losses_1079459212
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0` 2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ0` ::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0` 
 
_user_specified_nameinputs
®
¬
9__inference_batch_normalization_3_layer_call_fn_107948496

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall¾
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *]
fXRV
T__inference_batch_normalization_3_layer_call_and_return_conditional_losses_1079442402
StatefulPartitionedCall©
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ::::22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

±
T__inference_batch_normalization_2_layer_call_and_return_conditional_losses_107945278

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity¢AssignNewValue¢AssignNewValue_1t
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
!FusedBatchNormV3/ReadVariableOp_1Ø
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ0@:@:@:@:@:*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3ÿ
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
:ÿÿÿÿÿÿÿÿÿ0@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ0@::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@
 
_user_specified_nameinputs
Ì
±
T__inference_batch_normalization_7_layer_call_and_return_conditional_losses_107949119

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity¢AssignNewValue¢AssignNewValue_1t
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
!FusedBatchNormV3/ReadVariableOp_1ê
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ : : : : :*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3ÿ
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
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
®
¬
9__inference_batch_normalization_4_layer_call_fn_107948653

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall¾
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *]
fXRV
T__inference_batch_normalization_4_layer_call_and_return_conditional_losses_1079443562
StatefulPartitionedCall©
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ::::22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


T__inference_batch_normalization_7_layer_call_and_return_conditional_losses_107944812

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
!FusedBatchNormV3/ReadVariableOp_1Ü
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ : : : : :*
epsilon%o:*
is_training( 2
FusedBatchNormV3
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ :::::i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
Ê
¯
R__inference_batch_normalization_layer_call_and_return_conditional_losses_107943861

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity¢AssignNewValue¢AssignNewValue_1t
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
!FusedBatchNormV3/ReadVariableOp_1ê
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3ÿ
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
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
»
G
+__inference_re_lu_1_layer_call_fn_107948256

inputs
identityÑ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0` * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *O
fJRH
F__inference_re_lu_1_layer_call_and_return_conditional_losses_1079452242
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0` 2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ0` :W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0` 
 
_user_specified_nameinputs
Ó

T__inference_batch_normalization_1_layer_call_and_return_conditional_losses_107945183

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
!FusedBatchNormV3/ReadVariableOp_1Ê
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ0` : : : : :*
epsilon%o:*
is_training( 2
FusedBatchNormV3p
IdentityIdentityFusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0` 2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ0` :::::W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0` 
 
_user_specified_nameinputs
â
¬
9__inference_batch_normalization_2_layer_call_fn_107948403

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall«
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *]
fXRV
T__inference_batch_normalization_2_layer_call_and_return_conditional_losses_1079452962
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ0@::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@
 
_user_specified_nameinputs
¨

T__inference_batch_normalization_3_layer_call_and_return_conditional_losses_107948470

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
FusedBatchNormV3/ReadVariableOp®
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1á
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
is_training( 2
FusedBatchNormV3
IdentityIdentityFusedBatchNormV3:y:0*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ø
±
T__inference_batch_normalization_3_layer_call_and_return_conditional_losses_107944209

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity¢AssignNewValue¢AssignNewValue_1u
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
FusedBatchNormV3/ReadVariableOp®
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ï
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3ÿ
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
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


T__inference_batch_normalization_2_layer_call_and_return_conditional_losses_107944124

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
!FusedBatchNormV3/ReadVariableOp_1Ü
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@:@:@:@:@:*
epsilon%o:*
is_training( 2
FusedBatchNormV3
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@:::::i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
Ê
¯
R__inference_batch_normalization_layer_call_and_return_conditional_losses_107947981

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity¢AssignNewValue¢AssignNewValue_1t
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
!FusedBatchNormV3/ReadVariableOp_1ê
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3ÿ
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
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¨
¬
9__inference_batch_normalization_7_layer_call_fn_107949150

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall»
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *]
fXRV
T__inference_batch_normalization_7_layer_call_and_return_conditional_losses_1079447812
StatefulPartitionedCall¨
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ ::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs


*__inference_conv_6_layer_call_fn_107948759

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_conv_6_layer_call_and_return_conditional_losses_1079456022
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:ÿÿÿÿÿÿÿÿÿ::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
«
­
E__inference_conv_1_layer_call_and_return_conditional_losses_107947952

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp¤
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`À*
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
:ÿÿÿÿÿÿÿÿÿ`À2	
BiasAddm
IdentityIdentityBiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`À2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:ÿÿÿÿÿÿÿÿÿ`À:::X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`À
 
_user_specified_nameinputs
Ô
b
F__inference_re_lu_7_layer_call_and_return_conditional_losses_107945962

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0` 2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0` 2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ0` :W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0` 
 
_user_specified_nameinputs
¸
O
3__inference_max_pooling2d_1_layer_call_fn_107944031

inputs
identityô
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *W
fRRP
N__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_1079440252
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ó

T__inference_batch_normalization_7_layer_call_and_return_conditional_losses_107945921

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
!FusedBatchNormV3/ReadVariableOp_1Ê
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ0` : : : : :*
epsilon%o:*
is_training( 2
FusedBatchNormV3p
IdentityIdentityFusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0` 2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ0` :::::W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0` 
 
_user_specified_nameinputs

d
8__inference_tf_op_layer_concat_8_layer_call_fn_107949080
inputs_0
inputs_1
identityë
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0`@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *\
fWRU
S__inference_tf_op_layer_concat_8_layer_call_and_return_conditional_losses_1079458492
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0`@2

Identity"
identityIdentity:output:0*[
_input_shapesJ
H:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ0` :k g
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0` 
"
_user_specified_name
inputs/1
¿
G
+__inference_re_lu_8_layer_call_fn_107949407

inputs
identityÒ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`À* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *O
fJRH
F__inference_re_lu_8_layer_call_and_return_conditional_losses_1079460952
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`À2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ`À:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`À
 
_user_specified_nameinputs
Ô

.__inference_upsample_8_layer_call_fn_107944719

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *R
fMRK
I__inference_upsample_8_layer_call_and_return_conditional_losses_1079447092
StatefulPartitionedCall¨
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
Ô

.__inference_upsample_9_layer_call_fn_107944871

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *R
fMRK
I__inference_upsample_9_layer_call_and_return_conditional_losses_1079448612
StatefulPartitionedCall¨
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ ::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
¸
O
3__inference_max_pooling2d_3_layer_call_fn_107944263

inputs
identityô
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *W
fRRP
N__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_1079442572
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¦
­
E__inference_conv_3_layer_call_and_return_conditional_losses_107948266

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
:ÿÿÿÿÿÿÿÿÿ0@*
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
:ÿÿÿÿÿÿÿÿÿ0@2	
BiasAddl
IdentityIdentityBiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ0 :::W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0 
 
_user_specified_nameinputs
Ì
±
T__inference_batch_normalization_1_layer_call_and_return_conditional_losses_107943977

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity¢AssignNewValue¢AssignNewValue_1t
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
!FusedBatchNormV3/ReadVariableOp_1ê
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ : : : : :*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3ÿ
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
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs

±
T__inference_batch_normalization_3_layer_call_and_return_conditional_losses_107945391

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity¢AssignNewValue¢AssignNewValue_1u
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
FusedBatchNormV3/ReadVariableOp®
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ý
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3ÿ
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
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Í
à
0__inference_functional_1_layer_call_fn_107946919

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
identity¢StatefulPartitionedCallè	
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
:ÿÿÿÿÿÿÿÿÿ`À*b
_read_only_resource_inputsD
B@	
 !"#$%&'()*+,-./0123456789:;<=>?@*2
config_proto" 

CPU

GPU2 *0J 8 *T
fORM
K__inference_functional_1_layer_call_and_return_conditional_losses_1079467882
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`À2

Identity"
identityIdentity:output:0*±
_input_shapes
:ÿÿÿÿÿÿÿÿÿ`À::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:\ X
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`À
$
_user_specified_name
imageInput
×
S
7__inference_tf_op_layer_RealDiv_layer_call_fn_107947931

inputs
identityÞ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`À* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *[
fVRT
R__inference_tf_op_layer_RealDiv_layer_call_and_return_conditional_losses_1079449852
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`À2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ`À:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`À
 
_user_specified_nameinputs
ä
¬
9__inference_batch_normalization_4_layer_call_fn_107948704

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCallª
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *]
fXRV
T__inference_batch_normalization_4_layer_call_and_return_conditional_losses_1079455042
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿ::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


*__inference_conv_8_layer_call_fn_107949099

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0` *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_conv_8_layer_call_and_return_conditional_losses_1079458682
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0` 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ0`@::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0`@
 
_user_specified_nameinputs

±
T__inference_batch_normalization_2_layer_call_and_return_conditional_losses_107948359

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity¢AssignNewValue¢AssignNewValue_1t
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
!FusedBatchNormV3/ReadVariableOp_1Ø
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ0@:@:@:@:@:*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3ÿ
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
:ÿÿÿÿÿÿÿÿÿ0@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ0@::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@
 
_user_specified_nameinputs
¬
¬
9__inference_batch_normalization_4_layer_call_fn_107948640

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall¼
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *]
fXRV
T__inference_batch_normalization_4_layer_call_and_return_conditional_losses_1079443252
StatefulPartitionedCall©
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ::::22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¦
­
E__inference_conv_2_layer_call_and_return_conditional_losses_107945130

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
:ÿÿÿÿÿÿÿÿÿ0` *
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
:ÿÿÿÿÿÿÿÿÿ0` 2	
BiasAddl
IdentityIdentityBiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0` 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ0`:::W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0`
 
_user_specified_nameinputs
Ø
b
F__inference_re_lu_4_layer_call_and_return_conditional_losses_107948722

inputs
identityW
ReluReluinputs*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Reluo
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ø
b
F__inference_re_lu_4_layer_call_and_return_conditional_losses_107945563

inputs
identityW
ReluReluinputs*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Reluo
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
«
­
E__inference_conv_9_layer_call_and_return_conditional_losses_107946001

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOp¤
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`À*
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
:ÿÿÿÿÿÿÿÿÿ`À2	
BiasAddm
IdentityIdentityBiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`À2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:ÿÿÿÿÿÿÿÿÿ`À :::X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`À 
 
_user_specified_nameinputs

±
T__inference_batch_normalization_6_layer_call_and_return_conditional_losses_107949013

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity¢AssignNewValue¢AssignNewValue_1t
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
!FusedBatchNormV3/ReadVariableOp_1Ø
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ0@:@:@:@:@:*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3ÿ
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
:ÿÿÿÿÿÿÿÿÿ0@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ0@::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@
 
_user_specified_nameinputs
Ì
±
T__inference_batch_normalization_8_layer_call_and_return_conditional_losses_107949289

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity¢AssignNewValue¢AssignNewValue_1t
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
!FusedBatchNormV3/ReadVariableOp_1ê
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3ÿ
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
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ì
±
T__inference_batch_normalization_2_layer_call_and_return_conditional_losses_107948295

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity¢AssignNewValue¢AssignNewValue_1t
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
!FusedBatchNormV3/ReadVariableOp_1ê
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@:@:@:@:@:*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3ÿ
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
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
Æ
j
N__inference_tf_op_layer_Sub_layer_call_and_return_conditional_losses_107947937

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
:ÿÿÿÿÿÿÿÿÿ`À2
Subd
IdentityIdentitySub:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`À2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ`À:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`À
 
_user_specified_nameinputs
ä
¬
9__inference_batch_normalization_5_layer_call_fn_107948874

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCallª
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *]
fXRV
T__inference_batch_normalization_5_layer_call_and_return_conditional_losses_1079456372
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿ::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
×

T__inference_batch_normalization_8_layer_call_and_return_conditional_losses_107949371

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
!FusedBatchNormV3/ReadVariableOp_1Ë
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*L
_output_shapes:
8:ÿÿÿÿÿÿÿÿÿ`À:::::*
epsilon%o:*
is_training( 2
FusedBatchNormV3q
IdentityIdentityFusedBatchNormV3:y:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`À2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿ`À:::::X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`À
 
_user_specified_nameinputs


T__inference_batch_normalization_1_layer_call_and_return_conditional_losses_107944008

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
!FusedBatchNormV3/ReadVariableOp_1Ü
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ : : : : :*
epsilon%o:*
is_training( 2
FusedBatchNormV3
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ :::::i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
Ø
b
F__inference_re_lu_5_layer_call_and_return_conditional_losses_107945696

inputs
identityW
ReluReluinputs*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Reluo
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
à
ª
7__inference_batch_normalization_layer_call_fn_107948076

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall¨
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`À*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *[
fVRT
R__inference_batch_normalization_layer_call_and_return_conditional_losses_1079450522
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`À2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿ`À::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`À
 
_user_specified_nameinputs
Ó

T__inference_batch_normalization_1_layer_call_and_return_conditional_losses_107948220

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
!FusedBatchNormV3/ReadVariableOp_1Ê
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ0` : : : : :*
epsilon%o:*
is_training( 2
FusedBatchNormV3p
IdentityIdentityFusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0` 2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ0` :::::W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0` 
 
_user_specified_nameinputs
ª
¬
9__inference_batch_normalization_1_layer_call_fn_107948182

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall½
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ *&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *]
fXRV
T__inference_batch_normalization_1_layer_call_and_return_conditional_losses_1079440082
StatefulPartitionedCall¨
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ ::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
ÃË
«
K__inference_functional_1_layer_call_and_return_conditional_losses_107946304

imageinput
conv_1_107946135
conv_1_107946137!
batch_normalization_107946140!
batch_normalization_107946142!
batch_normalization_107946144!
batch_normalization_107946146
conv_2_107946151
conv_2_107946153#
batch_normalization_1_107946156#
batch_normalization_1_107946158#
batch_normalization_1_107946160#
batch_normalization_1_107946162
conv_3_107946167
conv_3_107946169#
batch_normalization_2_107946172#
batch_normalization_2_107946174#
batch_normalization_2_107946176#
batch_normalization_2_107946178
conv_4_107946183
conv_4_107946185#
batch_normalization_3_107946188#
batch_normalization_3_107946190#
batch_normalization_3_107946192#
batch_normalization_3_107946194
conv_5_107946199
conv_5_107946201#
batch_normalization_4_107946204#
batch_normalization_4_107946206#
batch_normalization_4_107946208#
batch_normalization_4_107946210
upsample_6_107946214
upsample_6_107946216
conv_6_107946220
conv_6_107946222#
batch_normalization_5_107946225#
batch_normalization_5_107946227#
batch_normalization_5_107946229#
batch_normalization_5_107946231
upsample_7_107946235
upsample_7_107946237
conv_7_107946241
conv_7_107946243#
batch_normalization_6_107946246#
batch_normalization_6_107946248#
batch_normalization_6_107946250#
batch_normalization_6_107946252
upsample_8_107946256
upsample_8_107946258
conv_8_107946262
conv_8_107946264#
batch_normalization_7_107946267#
batch_normalization_7_107946269#
batch_normalization_7_107946271#
batch_normalization_7_107946273
upsample_9_107946277
upsample_9_107946279
conv_9_107946283
conv_9_107946285#
batch_normalization_8_107946288#
batch_normalization_8_107946290#
batch_normalization_8_107946292#
batch_normalization_8_107946294
final_107946298
final_107946300
identity¢+batch_normalization/StatefulPartitionedCall¢-batch_normalization_1/StatefulPartitionedCall¢-batch_normalization_2/StatefulPartitionedCall¢-batch_normalization_3/StatefulPartitionedCall¢-batch_normalization_4/StatefulPartitionedCall¢-batch_normalization_5/StatefulPartitionedCall¢-batch_normalization_6/StatefulPartitionedCall¢-batch_normalization_7/StatefulPartitionedCall¢-batch_normalization_8/StatefulPartitionedCall¢conv_1/StatefulPartitionedCall¢conv_2/StatefulPartitionedCall¢conv_3/StatefulPartitionedCall¢conv_4/StatefulPartitionedCall¢conv_5/StatefulPartitionedCall¢conv_6/StatefulPartitionedCall¢conv_7/StatefulPartitionedCall¢conv_8/StatefulPartitionedCall¢conv_9/StatefulPartitionedCall¢final/StatefulPartitionedCall¢"upsample_6/StatefulPartitionedCall¢"upsample_7/StatefulPartitionedCall¢"upsample_8/StatefulPartitionedCall¢"upsample_9/StatefulPartitionedCall
#tf_op_layer_RealDiv/PartitionedCallPartitionedCall
imageinput*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`À* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *[
fVRT
R__inference_tf_op_layer_RealDiv_layer_call_and_return_conditional_losses_1079449852%
#tf_op_layer_RealDiv/PartitionedCall 
tf_op_layer_Sub/PartitionedCallPartitionedCall,tf_op_layer_RealDiv/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`À* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *W
fRRP
N__inference_tf_op_layer_Sub_layer_call_and_return_conditional_losses_1079449992!
tf_op_layer_Sub/PartitionedCallÃ
conv_1/StatefulPartitionedCallStatefulPartitionedCall(tf_op_layer_Sub/PartitionedCall:output:0conv_1_107946135conv_1_107946137*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`À*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_conv_1_layer_call_and_return_conditional_losses_1079450172 
conv_1/StatefulPartitionedCallÅ
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall'conv_1/StatefulPartitionedCall:output:0batch_normalization_107946140batch_normalization_107946142batch_normalization_107946144batch_normalization_107946146*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`À*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *[
fVRT
R__inference_batch_normalization_layer_call_and_return_conditional_losses_1079450702-
+batch_normalization/StatefulPartitionedCall
re_lu/PartitionedCallPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`À* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *M
fHRF
D__inference_re_lu_layer_call_and_return_conditional_losses_1079451112
re_lu/PartitionedCall
max_pooling2d/PartitionedCallPartitionedCallre_lu/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0`* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *U
fPRN
L__inference_max_pooling2d_layer_call_and_return_conditional_losses_1079439092
max_pooling2d/PartitionedCallÀ
conv_2/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0conv_2_107946151conv_2_107946153*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0` *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_conv_2_layer_call_and_return_conditional_losses_1079451302 
conv_2/StatefulPartitionedCallÒ
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall'conv_2/StatefulPartitionedCall:output:0batch_normalization_1_107946156batch_normalization_1_107946158batch_normalization_1_107946160batch_normalization_1_107946162*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0` *&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *]
fXRV
T__inference_batch_normalization_1_layer_call_and_return_conditional_losses_1079451832/
-batch_normalization_1/StatefulPartitionedCall
re_lu_1/PartitionedCallPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0` * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *O
fJRH
F__inference_re_lu_1_layer_call_and_return_conditional_losses_1079452242
re_lu_1/PartitionedCall
max_pooling2d_1/PartitionedCallPartitionedCall re_lu_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0 * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *W
fRRP
N__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_1079440252!
max_pooling2d_1/PartitionedCallÂ
conv_3/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_1/PartitionedCall:output:0conv_3_107946167conv_3_107946169*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_conv_3_layer_call_and_return_conditional_losses_1079452432 
conv_3/StatefulPartitionedCallÒ
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall'conv_3/StatefulPartitionedCall:output:0batch_normalization_2_107946172batch_normalization_2_107946174batch_normalization_2_107946176batch_normalization_2_107946178*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *]
fXRV
T__inference_batch_normalization_2_layer_call_and_return_conditional_losses_1079452962/
-batch_normalization_2/StatefulPartitionedCall
re_lu_2/PartitionedCallPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *O
fJRH
F__inference_re_lu_2_layer_call_and_return_conditional_losses_1079453372
re_lu_2/PartitionedCall
max_pooling2d_2/PartitionedCallPartitionedCall re_lu_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *W
fRRP
N__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_1079441412!
max_pooling2d_2/PartitionedCallÃ
conv_4/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_2/PartitionedCall:output:0conv_4_107946183conv_4_107946185*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_conv_4_layer_call_and_return_conditional_losses_1079453562 
conv_4/StatefulPartitionedCallÓ
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCall'conv_4/StatefulPartitionedCall:output:0batch_normalization_3_107946188batch_normalization_3_107946190batch_normalization_3_107946192batch_normalization_3_107946194*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *]
fXRV
T__inference_batch_normalization_3_layer_call_and_return_conditional_losses_1079454092/
-batch_normalization_3/StatefulPartitionedCall
re_lu_3/PartitionedCallPartitionedCall6batch_normalization_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *O
fJRH
F__inference_re_lu_3_layer_call_and_return_conditional_losses_1079454502
re_lu_3/PartitionedCall
max_pooling2d_3/PartitionedCallPartitionedCall re_lu_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *W
fRRP
N__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_1079442572!
max_pooling2d_3/PartitionedCallÃ
conv_5/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_3/PartitionedCall:output:0conv_5_107946199conv_5_107946201*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_conv_5_layer_call_and_return_conditional_losses_1079454692 
conv_5/StatefulPartitionedCallÓ
-batch_normalization_4/StatefulPartitionedCallStatefulPartitionedCall'conv_5/StatefulPartitionedCall:output:0batch_normalization_4_107946204batch_normalization_4_107946206batch_normalization_4_107946208batch_normalization_4_107946210*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *]
fXRV
T__inference_batch_normalization_4_layer_call_and_return_conditional_losses_1079455222/
-batch_normalization_4/StatefulPartitionedCall
re_lu_4/PartitionedCallPartitionedCall6batch_normalization_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *O
fJRH
F__inference_re_lu_4_layer_call_and_return_conditional_losses_1079455632
re_lu_4/PartitionedCallá
"upsample_6/StatefulPartitionedCallStatefulPartitionedCall re_lu_4/PartitionedCall:output:0upsample_6_107946214upsample_6_107946216*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *R
fMRK
I__inference_upsample_6_layer_call_and_return_conditional_losses_1079444052$
"upsample_6/StatefulPartitionedCallÑ
$tf_op_layer_concat_6/PartitionedCallPartitionedCall+upsample_6/StatefulPartitionedCall:output:0 re_lu_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *\
fWRU
S__inference_tf_op_layer_concat_6_layer_call_and_return_conditional_losses_1079455832&
$tf_op_layer_concat_6/PartitionedCallÈ
conv_6/StatefulPartitionedCallStatefulPartitionedCall-tf_op_layer_concat_6/PartitionedCall:output:0conv_6_107946220conv_6_107946222*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_conv_6_layer_call_and_return_conditional_losses_1079456022 
conv_6/StatefulPartitionedCallÓ
-batch_normalization_5/StatefulPartitionedCallStatefulPartitionedCall'conv_6/StatefulPartitionedCall:output:0batch_normalization_5_107946225batch_normalization_5_107946227batch_normalization_5_107946229batch_normalization_5_107946231*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *]
fXRV
T__inference_batch_normalization_5_layer_call_and_return_conditional_losses_1079456552/
-batch_normalization_5/StatefulPartitionedCall
re_lu_5/PartitionedCallPartitionedCall6batch_normalization_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *O
fJRH
F__inference_re_lu_5_layer_call_and_return_conditional_losses_1079456962
re_lu_5/PartitionedCallà
"upsample_7/StatefulPartitionedCallStatefulPartitionedCall re_lu_5/PartitionedCall:output:0upsample_7_107946235upsample_7_107946237*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *R
fMRK
I__inference_upsample_7_layer_call_and_return_conditional_losses_1079445572$
"upsample_7/StatefulPartitionedCallÑ
$tf_op_layer_concat_7/PartitionedCallPartitionedCall+upsample_7/StatefulPartitionedCall:output:0 re_lu_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *\
fWRU
S__inference_tf_op_layer_concat_7_layer_call_and_return_conditional_losses_1079457162&
$tf_op_layer_concat_7/PartitionedCallÇ
conv_7/StatefulPartitionedCallStatefulPartitionedCall-tf_op_layer_concat_7/PartitionedCall:output:0conv_7_107946241conv_7_107946243*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_conv_7_layer_call_and_return_conditional_losses_1079457352 
conv_7/StatefulPartitionedCallÒ
-batch_normalization_6/StatefulPartitionedCallStatefulPartitionedCall'conv_7/StatefulPartitionedCall:output:0batch_normalization_6_107946246batch_normalization_6_107946248batch_normalization_6_107946250batch_normalization_6_107946252*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *]
fXRV
T__inference_batch_normalization_6_layer_call_and_return_conditional_losses_1079457882/
-batch_normalization_6/StatefulPartitionedCall
re_lu_6/PartitionedCallPartitionedCall6batch_normalization_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *O
fJRH
F__inference_re_lu_6_layer_call_and_return_conditional_losses_1079458292
re_lu_6/PartitionedCallà
"upsample_8/StatefulPartitionedCallStatefulPartitionedCall re_lu_6/PartitionedCall:output:0upsample_8_107946256upsample_8_107946258*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *R
fMRK
I__inference_upsample_8_layer_call_and_return_conditional_losses_1079447092$
"upsample_8/StatefulPartitionedCallÐ
$tf_op_layer_concat_8/PartitionedCallPartitionedCall+upsample_8/StatefulPartitionedCall:output:0 re_lu_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0`@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *\
fWRU
S__inference_tf_op_layer_concat_8_layer_call_and_return_conditional_losses_1079458492&
$tf_op_layer_concat_8/PartitionedCallÇ
conv_8/StatefulPartitionedCallStatefulPartitionedCall-tf_op_layer_concat_8/PartitionedCall:output:0conv_8_107946262conv_8_107946264*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0` *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_conv_8_layer_call_and_return_conditional_losses_1079458682 
conv_8/StatefulPartitionedCallÒ
-batch_normalization_7/StatefulPartitionedCallStatefulPartitionedCall'conv_8/StatefulPartitionedCall:output:0batch_normalization_7_107946267batch_normalization_7_107946269batch_normalization_7_107946271batch_normalization_7_107946273*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0` *&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *]
fXRV
T__inference_batch_normalization_7_layer_call_and_return_conditional_losses_1079459212/
-batch_normalization_7/StatefulPartitionedCall
re_lu_7/PartitionedCallPartitionedCall6batch_normalization_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0` * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *O
fJRH
F__inference_re_lu_7_layer_call_and_return_conditional_losses_1079459622
re_lu_7/PartitionedCallà
"upsample_9/StatefulPartitionedCallStatefulPartitionedCall re_lu_7/PartitionedCall:output:0upsample_9_107946277upsample_9_107946279*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *R
fMRK
I__inference_upsample_9_layer_call_and_return_conditional_losses_1079448612$
"upsample_9/StatefulPartitionedCallÏ
$tf_op_layer_concat_9/PartitionedCallPartitionedCall+upsample_9/StatefulPartitionedCall:output:0re_lu/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`À * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *\
fWRU
S__inference_tf_op_layer_concat_9_layer_call_and_return_conditional_losses_1079459822&
$tf_op_layer_concat_9/PartitionedCallÈ
conv_9/StatefulPartitionedCallStatefulPartitionedCall-tf_op_layer_concat_9/PartitionedCall:output:0conv_9_107946283conv_9_107946285*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`À*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_conv_9_layer_call_and_return_conditional_losses_1079460012 
conv_9/StatefulPartitionedCallÓ
-batch_normalization_8/StatefulPartitionedCallStatefulPartitionedCall'conv_9/StatefulPartitionedCall:output:0batch_normalization_8_107946288batch_normalization_8_107946290batch_normalization_8_107946292batch_normalization_8_107946294*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`À*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *]
fXRV
T__inference_batch_normalization_8_layer_call_and_return_conditional_losses_1079460542/
-batch_normalization_8/StatefulPartitionedCall
re_lu_8/PartitionedCallPartitionedCall6batch_normalization_8/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`À* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *O
fJRH
F__inference_re_lu_8_layer_call_and_return_conditional_losses_1079460952
re_lu_8/PartitionedCall¶
final/StatefulPartitionedCallStatefulPartitionedCall re_lu_8/PartitionedCall:output:0final_107946298final_107946300*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`À*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *M
fHRF
D__inference_final_layer_call_and_return_conditional_losses_1079461132
final/StatefulPartitionedCall
IdentityIdentity&final/StatefulPartitionedCall:output:0,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall.^batch_normalization_3/StatefulPartitionedCall.^batch_normalization_4/StatefulPartitionedCall.^batch_normalization_5/StatefulPartitionedCall.^batch_normalization_6/StatefulPartitionedCall.^batch_normalization_7/StatefulPartitionedCall.^batch_normalization_8/StatefulPartitionedCall^conv_1/StatefulPartitionedCall^conv_2/StatefulPartitionedCall^conv_3/StatefulPartitionedCall^conv_4/StatefulPartitionedCall^conv_5/StatefulPartitionedCall^conv_6/StatefulPartitionedCall^conv_7/StatefulPartitionedCall^conv_8/StatefulPartitionedCall^conv_9/StatefulPartitionedCall^final/StatefulPartitionedCall#^upsample_6/StatefulPartitionedCall#^upsample_7/StatefulPartitionedCall#^upsample_8/StatefulPartitionedCall#^upsample_9/StatefulPartitionedCall*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`À2

Identity"
identityIdentity:output:0*±
_input_shapes
:ÿÿÿÿÿÿÿÿÿ`À::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::2Z
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
:ÿÿÿÿÿÿÿÿÿ`À
$
_user_specified_name
imageInput
Ó

T__inference_batch_normalization_2_layer_call_and_return_conditional_losses_107945296

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
!FusedBatchNormV3/ReadVariableOp_1Ê
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ0@:@:@:@:@:*
epsilon%o:*
is_training( 2
FusedBatchNormV3p
IdentityIdentityFusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ0@:::::W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@
 
_user_specified_nameinputs

j
N__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_107944257

inputs
identity­
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
2	
MaxPool
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
»
E
)__inference_re_lu_layer_call_fn_107948099

inputs
identityÐ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`À* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *M
fHRF
D__inference_re_lu_layer_call_and_return_conditional_losses_1079451112
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`À2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ`À:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`À
 
_user_specified_nameinputs

í#
%__inference__traced_restore_107949843
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
identity_65¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_10¢AssignVariableOp_11¢AssignVariableOp_12¢AssignVariableOp_13¢AssignVariableOp_14¢AssignVariableOp_15¢AssignVariableOp_16¢AssignVariableOp_17¢AssignVariableOp_18¢AssignVariableOp_19¢AssignVariableOp_2¢AssignVariableOp_20¢AssignVariableOp_21¢AssignVariableOp_22¢AssignVariableOp_23¢AssignVariableOp_24¢AssignVariableOp_25¢AssignVariableOp_26¢AssignVariableOp_27¢AssignVariableOp_28¢AssignVariableOp_29¢AssignVariableOp_3¢AssignVariableOp_30¢AssignVariableOp_31¢AssignVariableOp_32¢AssignVariableOp_33¢AssignVariableOp_34¢AssignVariableOp_35¢AssignVariableOp_36¢AssignVariableOp_37¢AssignVariableOp_38¢AssignVariableOp_39¢AssignVariableOp_4¢AssignVariableOp_40¢AssignVariableOp_41¢AssignVariableOp_42¢AssignVariableOp_43¢AssignVariableOp_44¢AssignVariableOp_45¢AssignVariableOp_46¢AssignVariableOp_47¢AssignVariableOp_48¢AssignVariableOp_49¢AssignVariableOp_5¢AssignVariableOp_50¢AssignVariableOp_51¢AssignVariableOp_52¢AssignVariableOp_53¢AssignVariableOp_54¢AssignVariableOp_55¢AssignVariableOp_56¢AssignVariableOp_57¢AssignVariableOp_58¢AssignVariableOp_59¢AssignVariableOp_6¢AssignVariableOp_60¢AssignVariableOp_61¢AssignVariableOp_62¢AssignVariableOp_63¢AssignVariableOp_7¢AssignVariableOp_8¢AssignVariableOp_9
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
RestoreV2/shape_and_slicesó
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

Identity_6¥
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
Identity_11Á
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
Identity_15¶
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
Identity_17Á
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
Identity_21¶
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
Identity_23Á
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
Identity_27¶
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
Identity_29Á
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
Identity_35¶
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
Identity_37Á
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
Identity_43¶
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
Identity_45Á
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
Identity_51¶
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
Identity_53Á
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
Identity_59¶
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
Identity_61Á
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
NoOpÞ
Identity_64Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_64Ñ
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
ª
¬
9__inference_batch_normalization_7_layer_call_fn_107949163

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall½
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ *&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *]
fXRV
T__inference_batch_normalization_7_layer_call_and_return_conditional_losses_1079448122
StatefulPartitionedCall¨
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ ::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
Ô
b
F__inference_re_lu_2_layer_call_and_return_conditional_losses_107948408

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ0@:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@
 
_user_specified_nameinputs
©
­
E__inference_conv_7_layer_call_and_return_conditional_losses_107945735

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
:ÿÿÿÿÿÿÿÿÿ0@*
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
:ÿÿÿÿÿÿÿÿÿ0@2	
BiasAddl
IdentityIdentityBiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:ÿÿÿÿÿÿÿÿÿ0:::X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0
 
_user_specified_nameinputs
Ì$
»
I__inference_upsample_7_layer_call_and_return_conditional_losses_107944557

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
strided_slice/stack_2â
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
strided_slice_1/stack_2ì
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
strided_slice_2/stack_2ì
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
strided_slice_3/stack_2ì
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_3´
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*'
_output_shapes
:@*
dtype02!
conv2d_transpose/ReadVariableOpñ
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@*
paddingVALID*
strides
2
conv2d_transpose
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp¤
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@2	
BiasAdd~
IdentityIdentityBiasAdd:output:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ß

T__inference_batch_normalization_4_layer_call_and_return_conditional_losses_107948691

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
FusedBatchNormV3/ReadVariableOp®
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ï
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
is_training( 2
FusedBatchNormV3q
IdentityIdentityFusedBatchNormV3:y:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿ:::::X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
À

S__inference_tf_op_layer_concat_8_layer_call_and_return_conditional_losses_107949074
inputs_0
inputs_1
identityi
concat_8/axisConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
concat_8/axis
concat_8ConcatV2inputs_0inputs_1concat_8/axis:output:0*
N*
T0*
_cloned(*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0`@2

concat_8m
IdentityIdentityconcat_8:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0`@2

Identity"
identityIdentity:output:0*[
_input_shapesJ
H:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ0` :k g
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0` 
"
_user_specified_name
inputs/1
¿
G
+__inference_re_lu_3_layer_call_fn_107948570

inputs
identityÒ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *O
fJRH
F__inference_re_lu_3_layer_call_and_return_conditional_losses_1079454502
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
É$
»
I__inference_upsample_8_layer_call_and_return_conditional_losses_107944709

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
strided_slice/stack_2â
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
strided_slice_1/stack_2ì
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
strided_slice_2/stack_2ì
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
strided_slice_3/stack_2ì
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
conv2d_transpose/ReadVariableOpñ
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ *
paddingVALID*
strides
2
conv2d_transpose
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp¤
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2	
BiasAdd~
IdentityIdentityBiasAdd:output:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@:::i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
¦
ª
7__inference_batch_normalization_layer_call_fn_107948025

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall»
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *[
fVRT
R__inference_batch_normalization_layer_call_and_return_conditional_losses_1079438922
StatefulPartitionedCall¨
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


R__inference_batch_normalization_layer_call_and_return_conditional_losses_107943892

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
!FusedBatchNormV3/ReadVariableOp_1Ü
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
is_training( 2
FusedBatchNormV3
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

±
T__inference_batch_normalization_5_layer_call_and_return_conditional_losses_107948843

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity¢AssignNewValue¢AssignNewValue_1u
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
FusedBatchNormV3/ReadVariableOp®
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ý
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3ÿ
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
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ï
O
3__inference_tf_op_layer_Sub_layer_call_fn_107947942

inputs
identityÚ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`À* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *W
fRRP
N__inference_tf_op_layer_Sub_layer_call_and_return_conditional_losses_1079449992
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`À2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ`À:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`À
 
_user_specified_nameinputs


*__inference_conv_7_layer_call_fn_107948929

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_conv_7_layer_call_and_return_conditional_losses_1079457352
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:ÿÿÿÿÿÿÿÿÿ0::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0
 
_user_specified_nameinputs
Õ

R__inference_batch_normalization_layer_call_and_return_conditional_losses_107945070

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
!FusedBatchNormV3/ReadVariableOp_1Ë
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*L
_output_shapes:
8:ÿÿÿÿÿÿÿÿÿ`À:::::*
epsilon%o:*
is_training( 2
FusedBatchNormV3q
IdentityIdentityFusedBatchNormV3:y:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`À2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿ`À:::::X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`À
 
_user_specified_nameinputs
ö
¸!
K__inference_functional_1_layer_call_and_return_conditional_losses_107947363

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
identity¢"batch_normalization/AssignNewValue¢$batch_normalization/AssignNewValue_1¢$batch_normalization_1/AssignNewValue¢&batch_normalization_1/AssignNewValue_1¢$batch_normalization_2/AssignNewValue¢&batch_normalization_2/AssignNewValue_1¢$batch_normalization_3/AssignNewValue¢&batch_normalization_3/AssignNewValue_1¢$batch_normalization_4/AssignNewValue¢&batch_normalization_4/AssignNewValue_1¢$batch_normalization_5/AssignNewValue¢&batch_normalization_5/AssignNewValue_1¢$batch_normalization_6/AssignNewValue¢&batch_normalization_6/AssignNewValue_1¢$batch_normalization_7/AssignNewValue¢&batch_normalization_7/AssignNewValue_1¢$batch_normalization_8/AssignNewValue¢&batch_normalization_8/AssignNewValue_1
tf_op_layer_RealDiv/RealDiv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ÿB2
tf_op_layer_RealDiv/RealDiv/y¿
tf_op_layer_RealDiv/RealDivRealDivinputs&tf_op_layer_RealDiv/RealDiv/y:output:0*
T0*
_cloned(*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`À2
tf_op_layer_RealDiv/RealDivs
tf_op_layer_Sub/Sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
tf_op_layer_Sub/Sub/y¼
tf_op_layer_Sub/SubSubtf_op_layer_RealDiv/RealDiv:z:0tf_op_layer_Sub/Sub/y:output:0*
T0*
_cloned(*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`À2
tf_op_layer_Sub/Subª
conv_1/Conv2D/ReadVariableOpReadVariableOp%conv_1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
conv_1/Conv2D/ReadVariableOpÊ
conv_1/Conv2DConv2Dtf_op_layer_Sub/Sub:z:0$conv_1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`À*
paddingSAME*
strides
2
conv_1/Conv2D¡
conv_1/BiasAdd/ReadVariableOpReadVariableOp&conv_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
conv_1/BiasAdd/ReadVariableOp¥
conv_1/BiasAddBiasAddconv_1/Conv2D:output:0%conv_1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`À2
conv_1/BiasAdd°
"batch_normalization/ReadVariableOpReadVariableOp+batch_normalization_readvariableop_resource*
_output_shapes
:*
dtype02$
"batch_normalization/ReadVariableOp¶
$batch_normalization/ReadVariableOp_1ReadVariableOp-batch_normalization_readvariableop_1_resource*
_output_shapes
:*
dtype02&
$batch_normalization/ReadVariableOp_1ã
3batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOp<batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype025
3batch_normalization/FusedBatchNormV3/ReadVariableOpé
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp>batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype027
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1â
$batch_normalization/FusedBatchNormV3FusedBatchNormV3conv_1/BiasAdd:output:0*batch_normalization/ReadVariableOp:value:0,batch_normalization/ReadVariableOp_1:value:0;batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0=batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*L
_output_shapes:
8:ÿÿÿÿÿÿÿÿÿ`À:::::*
epsilon%o:*
exponential_avg_factor%
×#<2&
$batch_normalization/FusedBatchNormV3÷
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
:ÿÿÿÿÿÿÿÿÿ`À2

re_lu/ReluÀ
max_pooling2d/MaxPoolMaxPoolre_lu/Relu:activations:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0`*
ksize
*
paddingVALID*
strides
2
max_pooling2d/MaxPoolª
conv_2/Conv2D/ReadVariableOpReadVariableOp%conv_2_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
conv_2/Conv2D/ReadVariableOpÐ
conv_2/Conv2DConv2Dmax_pooling2d/MaxPool:output:0$conv_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0` *
paddingSAME*
strides
2
conv_2/Conv2D¡
conv_2/BiasAdd/ReadVariableOpReadVariableOp&conv_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02
conv_2/BiasAdd/ReadVariableOp¤
conv_2/BiasAddBiasAddconv_2/Conv2D:output:0%conv_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0` 2
conv_2/BiasAdd¶
$batch_normalization_1/ReadVariableOpReadVariableOp-batch_normalization_1_readvariableop_resource*
_output_shapes
: *
dtype02&
$batch_normalization_1/ReadVariableOp¼
&batch_normalization_1/ReadVariableOp_1ReadVariableOp/batch_normalization_1_readvariableop_1_resource*
_output_shapes
: *
dtype02(
&batch_normalization_1/ReadVariableOp_1é
5batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype027
5batch_normalization_1/FusedBatchNormV3/ReadVariableOpï
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype029
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1í
&batch_normalization_1/FusedBatchNormV3FusedBatchNormV3conv_2/BiasAdd:output:0,batch_normalization_1/ReadVariableOp:value:0.batch_normalization_1/ReadVariableOp_1:value:0=batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ0` : : : : :*
epsilon%o:*
exponential_avg_factor%
×#<2(
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
:ÿÿÿÿÿÿÿÿÿ0` 2
re_lu_1/ReluÆ
max_pooling2d_1/MaxPoolMaxPoolre_lu_1/Relu:activations:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0 *
ksize
*
paddingVALID*
strides
2
max_pooling2d_1/MaxPoolª
conv_3/Conv2D/ReadVariableOpReadVariableOp%conv_3_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02
conv_3/Conv2D/ReadVariableOpÒ
conv_3/Conv2DConv2D max_pooling2d_1/MaxPool:output:0$conv_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@*
paddingSAME*
strides
2
conv_3/Conv2D¡
conv_3/BiasAdd/ReadVariableOpReadVariableOp&conv_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
conv_3/BiasAdd/ReadVariableOp¤
conv_3/BiasAddBiasAddconv_3/Conv2D:output:0%conv_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@2
conv_3/BiasAdd¶
$batch_normalization_2/ReadVariableOpReadVariableOp-batch_normalization_2_readvariableop_resource*
_output_shapes
:@*
dtype02&
$batch_normalization_2/ReadVariableOp¼
&batch_normalization_2/ReadVariableOp_1ReadVariableOp/batch_normalization_2_readvariableop_1_resource*
_output_shapes
:@*
dtype02(
&batch_normalization_2/ReadVariableOp_1é
5batch_normalization_2/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype027
5batch_normalization_2/FusedBatchNormV3/ReadVariableOpï
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype029
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1í
&batch_normalization_2/FusedBatchNormV3FusedBatchNormV3conv_3/BiasAdd:output:0,batch_normalization_2/ReadVariableOp:value:0.batch_normalization_2/ReadVariableOp_1:value:0=batch_normalization_2/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ0@:@:@:@:@:*
epsilon%o:*
exponential_avg_factor%
×#<2(
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
:ÿÿÿÿÿÿÿÿÿ0@2
re_lu_2/ReluÆ
max_pooling2d_2/MaxPoolMaxPoolre_lu_2/Relu:activations:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
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
conv_4/Conv2D/ReadVariableOpÓ
conv_4/Conv2DConv2D max_pooling2d_2/MaxPool:output:0$conv_4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
2
conv_4/Conv2D¢
conv_4/BiasAdd/ReadVariableOpReadVariableOp&conv_4_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
conv_4/BiasAdd/ReadVariableOp¥
conv_4/BiasAddBiasAddconv_4/Conv2D:output:0%conv_4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
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
&batch_normalization_3/ReadVariableOp_1ê
5batch_normalization_3/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_3_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype027
5batch_normalization_3/FusedBatchNormV3/ReadVariableOpð
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype029
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1ò
&batch_normalization_3/FusedBatchNormV3FusedBatchNormV3conv_4/BiasAdd:output:0,batch_normalization_3/ReadVariableOp:value:0.batch_normalization_3/ReadVariableOp_1:value:0=batch_normalization_3/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
exponential_avg_factor%
×#<2(
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
:ÿÿÿÿÿÿÿÿÿ2
re_lu_3/ReluÇ
max_pooling2d_3/MaxPoolMaxPoolre_lu_3/Relu:activations:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
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
conv_5/Conv2D/ReadVariableOpÓ
conv_5/Conv2DConv2D max_pooling2d_3/MaxPool:output:0$conv_5/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
2
conv_5/Conv2D¢
conv_5/BiasAdd/ReadVariableOpReadVariableOp&conv_5_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
conv_5/BiasAdd/ReadVariableOp¥
conv_5/BiasAddBiasAddconv_5/Conv2D:output:0%conv_5/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
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
&batch_normalization_4/ReadVariableOp_1ê
5batch_normalization_4/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_4_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype027
5batch_normalization_4/FusedBatchNormV3/ReadVariableOpð
7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype029
7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1ò
&batch_normalization_4/FusedBatchNormV3FusedBatchNormV3conv_5/BiasAdd:output:0,batch_normalization_4/ReadVariableOp:value:0.batch_normalization_4/ReadVariableOp_1:value:0=batch_normalization_4/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
exponential_avg_factor%
×#<2(
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
:ÿÿÿÿÿÿÿÿÿ2
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
 upsample_6/strided_slice/stack_2¤
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
upsample_6/stack/3Ô
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
"upsample_6/strided_slice_1/stack_2®
upsample_6/strided_slice_1StridedSliceupsample_6/stack:output:0)upsample_6/strided_slice_1/stack:output:0+upsample_6/strided_slice_1/stack_1:output:0+upsample_6/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
upsample_6/strided_slice_1Ö
*upsample_6/conv2d_transpose/ReadVariableOpReadVariableOp3upsample_6_conv2d_transpose_readvariableop_resource*(
_output_shapes
:*
dtype02,
*upsample_6/conv2d_transpose/ReadVariableOp 
upsample_6/conv2d_transposeConv2DBackpropInputupsample_6/stack:output:02upsample_6/conv2d_transpose/ReadVariableOp:value:0re_lu_4/Relu:activations:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides
2
upsample_6/conv2d_transpose®
!upsample_6/BiasAdd/ReadVariableOpReadVariableOp*upsample_6_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02#
!upsample_6/BiasAdd/ReadVariableOp¿
upsample_6/BiasAddBiasAdd$upsample_6/conv2d_transpose:output:0)upsample_6/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
upsample_6/BiasAdd
"tf_op_layer_concat_6/concat_6/axisConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2$
"tf_op_layer_concat_6/concat_6/axis
tf_op_layer_concat_6/concat_6ConcatV2upsample_6/BiasAdd:output:0re_lu_3/Relu:activations:0+tf_op_layer_concat_6/concat_6/axis:output:0*
N*
T0*
_cloned(*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
tf_op_layer_concat_6/concat_6¬
conv_6/Conv2D/ReadVariableOpReadVariableOp%conv_6_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02
conv_6/Conv2D/ReadVariableOpÙ
conv_6/Conv2DConv2D&tf_op_layer_concat_6/concat_6:output:0$conv_6/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
2
conv_6/Conv2D¢
conv_6/BiasAdd/ReadVariableOpReadVariableOp&conv_6_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
conv_6/BiasAdd/ReadVariableOp¥
conv_6/BiasAddBiasAddconv_6/Conv2D:output:0%conv_6/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
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
&batch_normalization_5/ReadVariableOp_1ê
5batch_normalization_5/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_5_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype027
5batch_normalization_5/FusedBatchNormV3/ReadVariableOpð
7batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype029
7batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1ò
&batch_normalization_5/FusedBatchNormV3FusedBatchNormV3conv_6/BiasAdd:output:0,batch_normalization_5/ReadVariableOp:value:0.batch_normalization_5/ReadVariableOp_1:value:0=batch_normalization_5/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
exponential_avg_factor%
×#<2(
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
:ÿÿÿÿÿÿÿÿÿ2
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
 upsample_7/strided_slice/stack_2¤
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
upsample_7/stack/3Ô
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
"upsample_7/strided_slice_1/stack_2®
upsample_7/strided_slice_1StridedSliceupsample_7/stack:output:0)upsample_7/strided_slice_1/stack:output:0+upsample_7/strided_slice_1/stack_1:output:0+upsample_7/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
upsample_7/strided_slice_1Õ
*upsample_7/conv2d_transpose/ReadVariableOpReadVariableOp3upsample_7_conv2d_transpose_readvariableop_resource*'
_output_shapes
:@*
dtype02,
*upsample_7/conv2d_transpose/ReadVariableOp
upsample_7/conv2d_transposeConv2DBackpropInputupsample_7/stack:output:02upsample_7/conv2d_transpose/ReadVariableOp:value:0re_lu_5/Relu:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@*
paddingVALID*
strides
2
upsample_7/conv2d_transpose­
!upsample_7/BiasAdd/ReadVariableOpReadVariableOp*upsample_7_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02#
!upsample_7/BiasAdd/ReadVariableOp¾
upsample_7/BiasAddBiasAdd$upsample_7/conv2d_transpose:output:0)upsample_7/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@2
upsample_7/BiasAdd
"tf_op_layer_concat_7/concat_7/axisConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2$
"tf_op_layer_concat_7/concat_7/axis
tf_op_layer_concat_7/concat_7ConcatV2upsample_7/BiasAdd:output:0re_lu_2/Relu:activations:0+tf_op_layer_concat_7/concat_7/axis:output:0*
N*
T0*
_cloned(*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ02
tf_op_layer_concat_7/concat_7«
conv_7/Conv2D/ReadVariableOpReadVariableOp%conv_7_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype02
conv_7/Conv2D/ReadVariableOpØ
conv_7/Conv2DConv2D&tf_op_layer_concat_7/concat_7:output:0$conv_7/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@*
paddingSAME*
strides
2
conv_7/Conv2D¡
conv_7/BiasAdd/ReadVariableOpReadVariableOp&conv_7_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
conv_7/BiasAdd/ReadVariableOp¤
conv_7/BiasAddBiasAddconv_7/Conv2D:output:0%conv_7/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@2
conv_7/BiasAdd¶
$batch_normalization_6/ReadVariableOpReadVariableOp-batch_normalization_6_readvariableop_resource*
_output_shapes
:@*
dtype02&
$batch_normalization_6/ReadVariableOp¼
&batch_normalization_6/ReadVariableOp_1ReadVariableOp/batch_normalization_6_readvariableop_1_resource*
_output_shapes
:@*
dtype02(
&batch_normalization_6/ReadVariableOp_1é
5batch_normalization_6/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_6_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype027
5batch_normalization_6/FusedBatchNormV3/ReadVariableOpï
7batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype029
7batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1í
&batch_normalization_6/FusedBatchNormV3FusedBatchNormV3conv_7/BiasAdd:output:0,batch_normalization_6/ReadVariableOp:value:0.batch_normalization_6/ReadVariableOp_1:value:0=batch_normalization_6/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ0@:@:@:@:@:*
epsilon%o:*
exponential_avg_factor%
×#<2(
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
:ÿÿÿÿÿÿÿÿÿ0@2
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
 upsample_8/strided_slice/stack_2¤
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
upsample_8/stack/3Ô
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
"upsample_8/strided_slice_1/stack_2®
upsample_8/strided_slice_1StridedSliceupsample_8/stack:output:0)upsample_8/strided_slice_1/stack:output:0+upsample_8/strided_slice_1/stack_1:output:0+upsample_8/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
upsample_8/strided_slice_1Ô
*upsample_8/conv2d_transpose/ReadVariableOpReadVariableOp3upsample_8_conv2d_transpose_readvariableop_resource*&
_output_shapes
: @*
dtype02,
*upsample_8/conv2d_transpose/ReadVariableOp
upsample_8/conv2d_transposeConv2DBackpropInputupsample_8/stack:output:02upsample_8/conv2d_transpose/ReadVariableOp:value:0re_lu_6/Relu:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0` *
paddingVALID*
strides
2
upsample_8/conv2d_transpose­
!upsample_8/BiasAdd/ReadVariableOpReadVariableOp*upsample_8_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02#
!upsample_8/BiasAdd/ReadVariableOp¾
upsample_8/BiasAddBiasAdd$upsample_8/conv2d_transpose:output:0)upsample_8/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0` 2
upsample_8/BiasAdd
"tf_op_layer_concat_8/concat_8/axisConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2$
"tf_op_layer_concat_8/concat_8/axis
tf_op_layer_concat_8/concat_8ConcatV2upsample_8/BiasAdd:output:0re_lu_1/Relu:activations:0+tf_op_layer_concat_8/concat_8/axis:output:0*
N*
T0*
_cloned(*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0`@2
tf_op_layer_concat_8/concat_8ª
conv_8/Conv2D/ReadVariableOpReadVariableOp%conv_8_conv2d_readvariableop_resource*&
_output_shapes
:@ *
dtype02
conv_8/Conv2D/ReadVariableOpØ
conv_8/Conv2DConv2D&tf_op_layer_concat_8/concat_8:output:0$conv_8/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0` *
paddingSAME*
strides
2
conv_8/Conv2D¡
conv_8/BiasAdd/ReadVariableOpReadVariableOp&conv_8_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02
conv_8/BiasAdd/ReadVariableOp¤
conv_8/BiasAddBiasAddconv_8/Conv2D:output:0%conv_8/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0` 2
conv_8/BiasAdd¶
$batch_normalization_7/ReadVariableOpReadVariableOp-batch_normalization_7_readvariableop_resource*
_output_shapes
: *
dtype02&
$batch_normalization_7/ReadVariableOp¼
&batch_normalization_7/ReadVariableOp_1ReadVariableOp/batch_normalization_7_readvariableop_1_resource*
_output_shapes
: *
dtype02(
&batch_normalization_7/ReadVariableOp_1é
5batch_normalization_7/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_7_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype027
5batch_normalization_7/FusedBatchNormV3/ReadVariableOpï
7batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype029
7batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1í
&batch_normalization_7/FusedBatchNormV3FusedBatchNormV3conv_8/BiasAdd:output:0,batch_normalization_7/ReadVariableOp:value:0.batch_normalization_7/ReadVariableOp_1:value:0=batch_normalization_7/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ0` : : : : :*
epsilon%o:*
exponential_avg_factor%
×#<2(
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
:ÿÿÿÿÿÿÿÿÿ0` 2
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
 upsample_9/strided_slice/stack_2¤
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
B :À2
upsample_9/stack/2j
upsample_9/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2
upsample_9/stack/3Ô
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
"upsample_9/strided_slice_1/stack_2®
upsample_9/strided_slice_1StridedSliceupsample_9/stack:output:0)upsample_9/strided_slice_1/stack:output:0+upsample_9/strided_slice_1/stack_1:output:0+upsample_9/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
upsample_9/strided_slice_1Ô
*upsample_9/conv2d_transpose/ReadVariableOpReadVariableOp3upsample_9_conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype02,
*upsample_9/conv2d_transpose/ReadVariableOp 
upsample_9/conv2d_transposeConv2DBackpropInputupsample_9/stack:output:02upsample_9/conv2d_transpose/ReadVariableOp:value:0re_lu_7/Relu:activations:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`À*
paddingVALID*
strides
2
upsample_9/conv2d_transpose­
!upsample_9/BiasAdd/ReadVariableOpReadVariableOp*upsample_9_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02#
!upsample_9/BiasAdd/ReadVariableOp¿
upsample_9/BiasAddBiasAdd$upsample_9/conv2d_transpose:output:0)upsample_9/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`À2
upsample_9/BiasAdd
"tf_op_layer_concat_9/concat_9/axisConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2$
"tf_op_layer_concat_9/concat_9/axis
tf_op_layer_concat_9/concat_9ConcatV2upsample_9/BiasAdd:output:0re_lu/Relu:activations:0+tf_op_layer_concat_9/concat_9/axis:output:0*
N*
T0*
_cloned(*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`À 2
tf_op_layer_concat_9/concat_9ª
conv_9/Conv2D/ReadVariableOpReadVariableOp%conv_9_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
conv_9/Conv2D/ReadVariableOpÙ
conv_9/Conv2DConv2D&tf_op_layer_concat_9/concat_9:output:0$conv_9/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`À*
paddingSAME*
strides
2
conv_9/Conv2D¡
conv_9/BiasAdd/ReadVariableOpReadVariableOp&conv_9_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
conv_9/BiasAdd/ReadVariableOp¥
conv_9/BiasAddBiasAddconv_9/Conv2D:output:0%conv_9/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`À2
conv_9/BiasAdd¶
$batch_normalization_8/ReadVariableOpReadVariableOp-batch_normalization_8_readvariableop_resource*
_output_shapes
:*
dtype02&
$batch_normalization_8/ReadVariableOp¼
&batch_normalization_8/ReadVariableOp_1ReadVariableOp/batch_normalization_8_readvariableop_1_resource*
_output_shapes
:*
dtype02(
&batch_normalization_8/ReadVariableOp_1é
5batch_normalization_8/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_8_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype027
5batch_normalization_8/FusedBatchNormV3/ReadVariableOpï
7batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype029
7batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1î
&batch_normalization_8/FusedBatchNormV3FusedBatchNormV3conv_9/BiasAdd:output:0,batch_normalization_8/ReadVariableOp:value:0.batch_normalization_8/ReadVariableOp_1:value:0=batch_normalization_8/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*L
_output_shapes:
8:ÿÿÿÿÿÿÿÿÿ`À:::::*
epsilon%o:*
exponential_avg_factor%
×#<2(
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
:ÿÿÿÿÿÿÿÿÿ`À2
re_lu_8/Relu§
final/Conv2D/ReadVariableOpReadVariableOp$final_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
final/Conv2D/ReadVariableOpÊ
final/Conv2DConv2Dre_lu_8/Relu:activations:0#final/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`À*
paddingSAME*
strides
2
final/Conv2D
final/BiasAdd/ReadVariableOpReadVariableOp%final_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
final/BiasAdd/ReadVariableOp¡
final/BiasAddBiasAddfinal/Conv2D:output:0$final/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`À2
final/BiasAdd¿
IdentityIdentityfinal/BiasAdd:output:0#^batch_normalization/AssignNewValue%^batch_normalization/AssignNewValue_1%^batch_normalization_1/AssignNewValue'^batch_normalization_1/AssignNewValue_1%^batch_normalization_2/AssignNewValue'^batch_normalization_2/AssignNewValue_1%^batch_normalization_3/AssignNewValue'^batch_normalization_3/AssignNewValue_1%^batch_normalization_4/AssignNewValue'^batch_normalization_4/AssignNewValue_1%^batch_normalization_5/AssignNewValue'^batch_normalization_5/AssignNewValue_1%^batch_normalization_6/AssignNewValue'^batch_normalization_6/AssignNewValue_1%^batch_normalization_7/AssignNewValue'^batch_normalization_7/AssignNewValue_1%^batch_normalization_8/AssignNewValue'^batch_normalization_8/AssignNewValue_1*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`À2

Identity"
identityIdentity:output:0*±
_input_shapes
:ÿÿÿÿÿÿÿÿÿ`À::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::2H
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
:ÿÿÿÿÿÿÿÿÿ`À
 
_user_specified_nameinputs
Ö
`
D__inference_re_lu_layer_call_and_return_conditional_losses_107948094

inputs
identityW
ReluReluinputs*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`À2
Reluo
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`À2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ`À:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`À
 
_user_specified_nameinputs
»
à
0__inference_functional_1_layer_call_fn_107946612

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
identity¢StatefulPartitionedCallÖ	
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
:ÿÿÿÿÿÿÿÿÿ`À*P
_read_only_resource_inputs2
0.	
 !"#$'()*+,/01234789:;<?@*2
config_proto" 

CPU

GPU2 *0J 8 *T
fORM
K__inference_functional_1_layer_call_and_return_conditional_losses_1079464812
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`À2

Identity"
identityIdentity:output:0*±
_input_shapes
:ÿÿÿÿÿÿÿÿÿ`À::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:\ X
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`À
$
_user_specified_name
imageInput
â
ª
7__inference_batch_normalization_layer_call_fn_107948089

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCallª
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`À*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *[
fVRT
R__inference_batch_normalization_layer_call_and_return_conditional_losses_1079450702
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`À2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿ`À::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`À
 
_user_specified_nameinputs
Ì
±
T__inference_batch_normalization_6_layer_call_and_return_conditional_losses_107948949

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity¢AssignNewValue¢AssignNewValue_1t
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
!FusedBatchNormV3/ReadVariableOp_1ê
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@:@:@:@:@:*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3ÿ
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
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
ß

T__inference_batch_normalization_5_layer_call_and_return_conditional_losses_107948861

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
FusedBatchNormV3/ReadVariableOp®
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ï
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
is_training( 2
FusedBatchNormV3q
IdentityIdentityFusedBatchNormV3:y:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿ:::::X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
©
­
E__inference_conv_7_layer_call_and_return_conditional_losses_107948920

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
:ÿÿÿÿÿÿÿÿÿ0@*
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
:ÿÿÿÿÿÿÿÿÿ0@2	
BiasAddl
IdentityIdentityBiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:ÿÿÿÿÿÿÿÿÿ0:::X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0
 
_user_specified_nameinputs
¤
ª
7__inference_batch_normalization_layer_call_fn_107948012

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall¹
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *[
fVRT
R__inference_batch_normalization_layer_call_and_return_conditional_losses_1079438612
StatefulPartitionedCall¨
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


*__inference_conv_3_layer_call_fn_107948275

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_conv_3_layer_call_and_return_conditional_losses_1079452432
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ0 ::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0 
 
_user_specified_nameinputs

~
)__inference_final_layer_call_fn_107949426

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`À*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *M
fHRF
D__inference_final_layer_call_and_return_conditional_losses_1079461132
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`À2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:ÿÿÿÿÿÿÿÿÿ`À::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`À
 
_user_specified_nameinputs


*__inference_conv_2_layer_call_fn_107948118

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0` *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_conv_2_layer_call_and_return_conditional_losses_1079451302
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0` 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ0`::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0`
 
_user_specified_nameinputs
Ö

.__inference_upsample_7_layer_call_fn_107944567

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *R
fMRK
I__inference_upsample_7_layer_call_and_return_conditional_losses_1079445572
StatefulPartitionedCall¨
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ::22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

d
8__inference_tf_op_layer_concat_6_layer_call_fn_107948740
inputs_0
inputs_1
identityì
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *\
fWRU
S__inference_tf_op_layer_concat_6_layer_call_and_return_conditional_losses_1079455832
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*]
_input_shapesL
J:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:l h
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0:ZV
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/1
¦
­
E__inference_conv_8_layer_call_and_return_conditional_losses_107949090

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
:ÿÿÿÿÿÿÿÿÿ0` *
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
:ÿÿÿÿÿÿÿÿÿ0` 2	
BiasAddl
IdentityIdentityBiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0` 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ0`@:::W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0`@
 
_user_specified_nameinputs
à
¬
9__inference_batch_normalization_7_layer_call_fn_107949214

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall©
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0` *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *]
fXRV
T__inference_batch_normalization_7_layer_call_and_return_conditional_losses_1079459032
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0` 2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ0` ::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0` 
 
_user_specified_nameinputs
Ò$
»
I__inference_upsample_6_layer_call_and_return_conditional_losses_107944405

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
strided_slice/stack_2â
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
strided_slice_1/stack_2ì
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
strided_slice_2/stack_2ì
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
strided_slice_3/stack_2ì
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_3µ
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*(
_output_shapes
:*
dtype02!
conv2d_transpose/ReadVariableOpò
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides
2
conv2d_transpose
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp¥
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2	
BiasAdd
IdentityIdentityBiasAdd:output:0*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
à
¬
9__inference_batch_normalization_6_layer_call_fn_107949044

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall©
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *]
fXRV
T__inference_batch_normalization_6_layer_call_and_return_conditional_losses_1079457702
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ0@::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@
 
_user_specified_nameinputs
æ
¬
9__inference_batch_normalization_4_layer_call_fn_107948717

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall¬
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *]
fXRV
T__inference_batch_normalization_4_layer_call_and_return_conditional_losses_1079455222
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿ::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ô
b
F__inference_re_lu_6_layer_call_and_return_conditional_losses_107945829

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ0@:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0@
 
_user_specified_nameinputs

d
8__inference_tf_op_layer_concat_9_layer_call_fn_107949250
inputs_0
inputs_1
identityì
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`À * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *\
fWRU
S__inference_tf_op_layer_concat_9_layer_call_and_return_conditional_losses_1079459822
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`À 2

Identity"
identityIdentity:output:0*\
_input_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ`À:k g
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0:ZV
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`À
"
_user_specified_name
inputs/1
¨

T__inference_batch_normalization_4_layer_call_and_return_conditional_losses_107944356

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
FusedBatchNormV3/ReadVariableOp®
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1á
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
is_training( 2
FusedBatchNormV3
IdentityIdentityFusedBatchNormV3:y:0*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

±
T__inference_batch_normalization_1_layer_call_and_return_conditional_losses_107945165

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity¢AssignNewValue¢AssignNewValue_1t
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
!FusedBatchNormV3/ReadVariableOp_1Ø
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ0` : : : : :*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3ÿ
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
:ÿÿÿÿÿÿÿÿÿ0` 2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ0` ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0` 
 
_user_specified_nameinputs
«
­
E__inference_conv_1_layer_call_and_return_conditional_losses_107945017

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp¤
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`À*
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
:ÿÿÿÿÿÿÿÿÿ`À2	
BiasAddm
IdentityIdentityBiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`À2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:ÿÿÿÿÿÿÿÿÿ`À:::X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`À
 
_user_specified_nameinputs


T__inference_batch_normalization_6_layer_call_and_return_conditional_losses_107948967

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
!FusedBatchNormV3/ReadVariableOp_1Ü
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@:@:@:@:@:*
epsilon%o:*
is_training( 2
FusedBatchNormV3
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@:::::i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs"¸L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*À
serving_default¬
J

imageInput<
serving_default_imageInput:0ÿÿÿÿÿÿÿÿÿ`ÀB
final9
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿ`Àtensorflow/serving/predict:Ô

½í
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
ù__call__
+ú&call_and_return_all_conditional_losses
û_default_save_signature"Êâ
_tf_keras_network­â{"class_name": "Functional", "name": "functional_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "functional_1", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 96, 192, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "imageInput"}, "name": "imageInput", "inbound_nodes": []}, {"class_name": "TensorFlowOpLayer", "config": {"name": "RealDiv", "trainable": true, "dtype": "float32", "node_def": {"name": "RealDiv", "op": "RealDiv", "input": ["imageInput", "RealDiv/y"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {"1": 127.5}}, "name": "tf_op_layer_RealDiv", "inbound_nodes": [[["imageInput", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "Sub", "trainable": true, "dtype": "float32", "node_def": {"name": "Sub", "op": "Sub", "input": ["RealDiv", "Sub/y"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {"1": 1.0}}, "name": "tf_op_layer_Sub", "inbound_nodes": [[["tf_op_layer_RealDiv", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv_1", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv_1", "inbound_nodes": [[["tf_op_layer_Sub", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization", "inbound_nodes": [[["conv_1", 0, 0, {}]]]}, {"class_name": "ReLU", "config": {"name": "re_lu", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}, "name": "re_lu", "inbound_nodes": [[["batch_normalization", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "max_pooling2d", "inbound_nodes": [[["re_lu", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv_2", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv_2", "inbound_nodes": [[["max_pooling2d", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_1", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_1", "inbound_nodes": [[["conv_2", 0, 0, {}]]]}, {"class_name": "ReLU", "config": {"name": "re_lu_1", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}, "name": "re_lu_1", "inbound_nodes": [[["batch_normalization_1", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_1", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "max_pooling2d_1", "inbound_nodes": [[["re_lu_1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv_3", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv_3", "inbound_nodes": [[["max_pooling2d_1", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_2", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_2", "inbound_nodes": [[["conv_3", 0, 0, {}]]]}, {"class_name": "ReLU", "config": {"name": "re_lu_2", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}, "name": "re_lu_2", "inbound_nodes": [[["batch_normalization_2", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_2", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "max_pooling2d_2", "inbound_nodes": [[["re_lu_2", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv_4", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv_4", "inbound_nodes": [[["max_pooling2d_2", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_3", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_3", "inbound_nodes": [[["conv_4", 0, 0, {}]]]}, {"class_name": "ReLU", "config": {"name": "re_lu_3", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}, "name": "re_lu_3", "inbound_nodes": [[["batch_normalization_3", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_3", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "max_pooling2d_3", "inbound_nodes": [[["re_lu_3", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv_5", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv_5", "inbound_nodes": [[["max_pooling2d_3", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_4", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_4", "inbound_nodes": [[["conv_5", 0, 0, {}]]]}, {"class_name": "ReLU", "config": {"name": "re_lu_4", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}, "name": "re_lu_4", "inbound_nodes": [[["batch_normalization_4", 0, 0, {}]]]}, {"class_name": "Conv2DTranspose", "config": {"name": "upsample_6", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [2, 2]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "name": "upsample_6", "inbound_nodes": [[["re_lu_4", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "concat_6", "trainable": true, "dtype": "float32", "node_def": {"name": "concat_6", "op": "ConcatV2", "input": ["upsample_6/BiasAdd", "re_lu_3/Relu", "concat_6/axis"], "attr": {"Tidx": {"type": "DT_INT32"}, "T": {"type": "DT_FLOAT"}, "N": {"i": "2"}}}, "constants": {"2": -1}}, "name": "tf_op_layer_concat_6", "inbound_nodes": [[["upsample_6", 0, 0, {}], ["re_lu_3", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv_6", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv_6", "inbound_nodes": [[["tf_op_layer_concat_6", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_5", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_5", "inbound_nodes": [[["conv_6", 0, 0, {}]]]}, {"class_name": "ReLU", "config": {"name": "re_lu_5", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}, "name": "re_lu_5", "inbound_nodes": [[["batch_normalization_5", 0, 0, {}]]]}, {"class_name": "Conv2DTranspose", "config": {"name": "upsample_7", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [2, 2]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "name": "upsample_7", "inbound_nodes": [[["re_lu_5", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "concat_7", "trainable": true, "dtype": "float32", "node_def": {"name": "concat_7", "op": "ConcatV2", "input": ["upsample_7/BiasAdd", "re_lu_2/Relu", "concat_7/axis"], "attr": {"Tidx": {"type": "DT_INT32"}, "T": {"type": "DT_FLOAT"}, "N": {"i": "2"}}}, "constants": {"2": -1}}, "name": "tf_op_layer_concat_7", "inbound_nodes": [[["upsample_7", 0, 0, {}], ["re_lu_2", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv_7", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv_7", "inbound_nodes": [[["tf_op_layer_concat_7", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_6", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_6", "inbound_nodes": [[["conv_7", 0, 0, {}]]]}, {"class_name": "ReLU", "config": {"name": "re_lu_6", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}, "name": "re_lu_6", "inbound_nodes": [[["batch_normalization_6", 0, 0, {}]]]}, {"class_name": "Conv2DTranspose", "config": {"name": "upsample_8", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [2, 2]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "name": "upsample_8", "inbound_nodes": [[["re_lu_6", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "concat_8", "trainable": true, "dtype": "float32", "node_def": {"name": "concat_8", "op": "ConcatV2", "input": ["upsample_8/BiasAdd", "re_lu_1/Relu", "concat_8/axis"], "attr": {"Tidx": {"type": "DT_INT32"}, "T": {"type": "DT_FLOAT"}, "N": {"i": "2"}}}, "constants": {"2": -1}}, "name": "tf_op_layer_concat_8", "inbound_nodes": [[["upsample_8", 0, 0, {}], ["re_lu_1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv_8", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv_8", "inbound_nodes": [[["tf_op_layer_concat_8", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_7", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_7", "inbound_nodes": [[["conv_8", 0, 0, {}]]]}, {"class_name": "ReLU", "config": {"name": "re_lu_7", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}, "name": "re_lu_7", "inbound_nodes": [[["batch_normalization_7", 0, 0, {}]]]}, {"class_name": "Conv2DTranspose", "config": {"name": "upsample_9", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [2, 2]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "name": "upsample_9", "inbound_nodes": [[["re_lu_7", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "concat_9", "trainable": true, "dtype": "float32", "node_def": {"name": "concat_9", "op": "ConcatV2", "input": ["upsample_9/BiasAdd", "re_lu/Relu", "concat_9/axis"], "attr": {"Tidx": {"type": "DT_INT32"}, "T": {"type": "DT_FLOAT"}, "N": {"i": "2"}}}, "constants": {"2": -1}}, "name": "tf_op_layer_concat_9", "inbound_nodes": [[["upsample_9", 0, 0, {}], ["re_lu", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv_9", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv_9", "inbound_nodes": [[["tf_op_layer_concat_9", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_8", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_8", "inbound_nodes": [[["conv_9", 0, 0, {}]]]}, {"class_name": "ReLU", "config": {"name": "re_lu_8", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}, "name": "re_lu_8", "inbound_nodes": [[["batch_normalization_8", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "final", "trainable": true, "dtype": "float32", "filters": 2, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "final", "inbound_nodes": [[["re_lu_8", 0, 0, {}]]]}], "input_layers": [["imageInput", 0, 0]], "output_layers": [["final", 0, 0]]}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 96, 192, 3]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "functional_1", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 96, 192, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "imageInput"}, "name": "imageInput", "inbound_nodes": []}, {"class_name": "TensorFlowOpLayer", "config": {"name": "RealDiv", "trainable": true, "dtype": "float32", "node_def": {"name": "RealDiv", "op": "RealDiv", "input": ["imageInput", "RealDiv/y"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {"1": 127.5}}, "name": "tf_op_layer_RealDiv", "inbound_nodes": [[["imageInput", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "Sub", "trainable": true, "dtype": "float32", "node_def": {"name": "Sub", "op": "Sub", "input": ["RealDiv", "Sub/y"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {"1": 1.0}}, "name": "tf_op_layer_Sub", "inbound_nodes": [[["tf_op_layer_RealDiv", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv_1", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv_1", "inbound_nodes": [[["tf_op_layer_Sub", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization", "inbound_nodes": [[["conv_1", 0, 0, {}]]]}, {"class_name": "ReLU", "config": {"name": "re_lu", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}, "name": "re_lu", "inbound_nodes": [[["batch_normalization", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "max_pooling2d", "inbound_nodes": [[["re_lu", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv_2", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv_2", "inbound_nodes": [[["max_pooling2d", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_1", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_1", "inbound_nodes": [[["conv_2", 0, 0, {}]]]}, {"class_name": "ReLU", "config": {"name": "re_lu_1", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}, "name": "re_lu_1", "inbound_nodes": [[["batch_normalization_1", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_1", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "max_pooling2d_1", "inbound_nodes": [[["re_lu_1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv_3", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv_3", "inbound_nodes": [[["max_pooling2d_1", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_2", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_2", "inbound_nodes": [[["conv_3", 0, 0, {}]]]}, {"class_name": "ReLU", "config": {"name": "re_lu_2", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}, "name": "re_lu_2", "inbound_nodes": [[["batch_normalization_2", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_2", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "max_pooling2d_2", "inbound_nodes": [[["re_lu_2", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv_4", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv_4", "inbound_nodes": [[["max_pooling2d_2", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_3", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_3", "inbound_nodes": [[["conv_4", 0, 0, {}]]]}, {"class_name": "ReLU", "config": {"name": "re_lu_3", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}, "name": "re_lu_3", "inbound_nodes": [[["batch_normalization_3", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_3", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "max_pooling2d_3", "inbound_nodes": [[["re_lu_3", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv_5", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv_5", "inbound_nodes": [[["max_pooling2d_3", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_4", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_4", "inbound_nodes": [[["conv_5", 0, 0, {}]]]}, {"class_name": "ReLU", "config": {"name": "re_lu_4", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}, "name": "re_lu_4", "inbound_nodes": [[["batch_normalization_4", 0, 0, {}]]]}, {"class_name": "Conv2DTranspose", "config": {"name": "upsample_6", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [2, 2]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "name": "upsample_6", "inbound_nodes": [[["re_lu_4", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "concat_6", "trainable": true, "dtype": "float32", "node_def": {"name": "concat_6", "op": "ConcatV2", "input": ["upsample_6/BiasAdd", "re_lu_3/Relu", "concat_6/axis"], "attr": {"Tidx": {"type": "DT_INT32"}, "T": {"type": "DT_FLOAT"}, "N": {"i": "2"}}}, "constants": {"2": -1}}, "name": "tf_op_layer_concat_6", "inbound_nodes": [[["upsample_6", 0, 0, {}], ["re_lu_3", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv_6", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv_6", "inbound_nodes": [[["tf_op_layer_concat_6", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_5", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_5", "inbound_nodes": [[["conv_6", 0, 0, {}]]]}, {"class_name": "ReLU", "config": {"name": "re_lu_5", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}, "name": "re_lu_5", "inbound_nodes": [[["batch_normalization_5", 0, 0, {}]]]}, {"class_name": "Conv2DTranspose", "config": {"name": "upsample_7", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [2, 2]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "name": "upsample_7", "inbound_nodes": [[["re_lu_5", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "concat_7", "trainable": true, "dtype": "float32", "node_def": {"name": "concat_7", "op": "ConcatV2", "input": ["upsample_7/BiasAdd", "re_lu_2/Relu", "concat_7/axis"], "attr": {"Tidx": {"type": "DT_INT32"}, "T": {"type": "DT_FLOAT"}, "N": {"i": "2"}}}, "constants": {"2": -1}}, "name": "tf_op_layer_concat_7", "inbound_nodes": [[["upsample_7", 0, 0, {}], ["re_lu_2", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv_7", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv_7", "inbound_nodes": [[["tf_op_layer_concat_7", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_6", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_6", "inbound_nodes": [[["conv_7", 0, 0, {}]]]}, {"class_name": "ReLU", "config": {"name": "re_lu_6", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}, "name": "re_lu_6", "inbound_nodes": [[["batch_normalization_6", 0, 0, {}]]]}, {"class_name": "Conv2DTranspose", "config": {"name": "upsample_8", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [2, 2]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "name": "upsample_8", "inbound_nodes": [[["re_lu_6", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "concat_8", "trainable": true, "dtype": "float32", "node_def": {"name": "concat_8", "op": "ConcatV2", "input": ["upsample_8/BiasAdd", "re_lu_1/Relu", "concat_8/axis"], "attr": {"Tidx": {"type": "DT_INT32"}, "T": {"type": "DT_FLOAT"}, "N": {"i": "2"}}}, "constants": {"2": -1}}, "name": "tf_op_layer_concat_8", "inbound_nodes": [[["upsample_8", 0, 0, {}], ["re_lu_1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv_8", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv_8", "inbound_nodes": [[["tf_op_layer_concat_8", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_7", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_7", "inbound_nodes": [[["conv_8", 0, 0, {}]]]}, {"class_name": "ReLU", "config": {"name": "re_lu_7", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}, "name": "re_lu_7", "inbound_nodes": [[["batch_normalization_7", 0, 0, {}]]]}, {"class_name": "Conv2DTranspose", "config": {"name": "upsample_9", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [2, 2]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "name": "upsample_9", "inbound_nodes": [[["re_lu_7", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "concat_9", "trainable": true, "dtype": "float32", "node_def": {"name": "concat_9", "op": "ConcatV2", "input": ["upsample_9/BiasAdd", "re_lu/Relu", "concat_9/axis"], "attr": {"Tidx": {"type": "DT_INT32"}, "T": {"type": "DT_FLOAT"}, "N": {"i": "2"}}}, "constants": {"2": -1}}, "name": "tf_op_layer_concat_9", "inbound_nodes": [[["upsample_9", 0, 0, {}], ["re_lu", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv_9", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv_9", "inbound_nodes": [[["tf_op_layer_concat_9", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_8", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_8", "inbound_nodes": [[["conv_9", 0, 0, {}]]]}, {"class_name": "ReLU", "config": {"name": "re_lu_8", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}, "name": "re_lu_8", "inbound_nodes": [[["batch_normalization_8", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "final", "trainable": true, "dtype": "float32", "filters": 2, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "final", "inbound_nodes": [[["re_lu_8", 0, 0, {}]]]}], "input_layers": [["imageInput", 0, 0]], "output_layers": [["final", 0, 0]]}}}
"þ
_tf_keras_input_layerÞ{"class_name": "InputLayer", "name": "imageInput", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 96, 192, 3]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 96, 192, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "imageInput"}}
â
1regularization_losses
2	variables
3trainable_variables
4	keras_api
ü__call__
+ý&call_and_return_all_conditional_losses"Ñ
_tf_keras_layer·{"class_name": "TensorFlowOpLayer", "name": "tf_op_layer_RealDiv", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "RealDiv", "trainable": true, "dtype": "float32", "node_def": {"name": "RealDiv", "op": "RealDiv", "input": ["imageInput", "RealDiv/y"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {"1": 127.5}}}
É
5regularization_losses
6	variables
7trainable_variables
8	keras_api
þ__call__
+ÿ&call_and_return_all_conditional_losses"¸
_tf_keras_layer{"class_name": "TensorFlowOpLayer", "name": "tf_op_layer_Sub", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "Sub", "trainable": true, "dtype": "float32", "node_def": {"name": "Sub", "op": "Sub", "input": ["RealDiv", "Sub/y"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {"1": 1.0}}}
ñ	

9kernel
:bias
;regularization_losses
<	variables
=trainable_variables
>	keras_api
__call__
+&call_and_return_all_conditional_losses"Ê
_tf_keras_layer°{"class_name": "Conv2D", "name": "conv_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv_1", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 3}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 96, 192, 3]}}
¹	
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
+&call_and_return_all_conditional_losses"ã
_tf_keras_layerÉ{"class_name": "BatchNormalization", "name": "batch_normalization", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 16}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 96, 192, 16]}}
é
Hregularization_losses
I	variables
Jtrainable_variables
K	keras_api
__call__
+&call_and_return_all_conditional_losses"Ø
_tf_keras_layer¾{"class_name": "ReLU", "name": "re_lu", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "re_lu", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}}
ý
Lregularization_losses
M	variables
Ntrainable_variables
O	keras_api
__call__
+&call_and_return_all_conditional_losses"ì
_tf_keras_layerÒ{"class_name": "MaxPooling2D", "name": "max_pooling2d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "max_pooling2d", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
ò	

Pkernel
Qbias
Rregularization_losses
S	variables
Ttrainable_variables
U	keras_api
__call__
+&call_and_return_all_conditional_losses"Ë
_tf_keras_layer±{"class_name": "Conv2D", "name": "conv_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv_2", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 16}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 48, 96, 16]}}
¼	
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
+&call_and_return_all_conditional_losses"æ
_tf_keras_layerÌ{"class_name": "BatchNormalization", "name": "batch_normalization_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_1", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 48, 96, 32]}}
í
_regularization_losses
`	variables
atrainable_variables
b	keras_api
__call__
+&call_and_return_all_conditional_losses"Ü
_tf_keras_layerÂ{"class_name": "ReLU", "name": "re_lu_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "re_lu_1", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}}

cregularization_losses
d	variables
etrainable_variables
f	keras_api
__call__
+&call_and_return_all_conditional_losses"ð
_tf_keras_layerÖ{"class_name": "MaxPooling2D", "name": "max_pooling2d_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "max_pooling2d_1", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
ò	

gkernel
hbias
iregularization_losses
j	variables
ktrainable_variables
l	keras_api
__call__
+&call_and_return_all_conditional_losses"Ë
_tf_keras_layer±{"class_name": "Conv2D", "name": "conv_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv_3", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 24, 48, 32]}}
¼	
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
+&call_and_return_all_conditional_losses"æ
_tf_keras_layerÌ{"class_name": "BatchNormalization", "name": "batch_normalization_2", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_2", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 24, 48, 64]}}
í
vregularization_losses
w	variables
xtrainable_variables
y	keras_api
__call__
+&call_and_return_all_conditional_losses"Ü
_tf_keras_layerÂ{"class_name": "ReLU", "name": "re_lu_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "re_lu_2", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}}

zregularization_losses
{	variables
|trainable_variables
}	keras_api
__call__
+&call_and_return_all_conditional_losses"ð
_tf_keras_layerÖ{"class_name": "MaxPooling2D", "name": "max_pooling2d_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "max_pooling2d_2", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
÷	

~kernel
bias
regularization_losses
	variables
trainable_variables
	keras_api
__call__
+&call_and_return_all_conditional_losses"Ì
_tf_keras_layer²{"class_name": "Conv2D", "name": "conv_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv_4", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 12, 24, 64]}}
Ç	
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
+&call_and_return_all_conditional_losses"è
_tf_keras_layerÎ{"class_name": "BatchNormalization", "name": "batch_normalization_3", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_3", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 12, 24, 128]}}
ñ
regularization_losses
	variables
trainable_variables
	keras_api
__call__
+&call_and_return_all_conditional_losses"Ü
_tf_keras_layerÂ{"class_name": "ReLU", "name": "re_lu_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "re_lu_3", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}}

regularization_losses
	variables
trainable_variables
	keras_api
__call__
+&call_and_return_all_conditional_losses"ð
_tf_keras_layerÖ{"class_name": "MaxPooling2D", "name": "max_pooling2d_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "max_pooling2d_3", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
ú	
kernel
	bias
regularization_losses
	variables
trainable_variables
	keras_api
 __call__
+¡&call_and_return_all_conditional_losses"Í
_tf_keras_layer³{"class_name": "Conv2D", "name": "conv_5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv_5", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 6, 12, 128]}}
Æ	
	axis

gamma
	beta
moving_mean
moving_variance
 regularization_losses
¡	variables
¢trainable_variables
£	keras_api
¢__call__
+£&call_and_return_all_conditional_losses"ç
_tf_keras_layerÍ{"class_name": "BatchNormalization", "name": "batch_normalization_4", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_4", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 256}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 6, 12, 256]}}
ñ
¤regularization_losses
¥	variables
¦trainable_variables
§	keras_api
¤__call__
+¥&call_and_return_all_conditional_losses"Ü
_tf_keras_layerÂ{"class_name": "ReLU", "name": "re_lu_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "re_lu_4", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}}
¤

¨kernel
	©bias
ªregularization_losses
«	variables
¬trainable_variables
­	keras_api
¦__call__
+§&call_and_return_all_conditional_losses"÷
_tf_keras_layerÝ{"class_name": "Conv2DTranspose", "name": "upsample_6", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "upsample_6", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [2, 2]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 256}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 6, 12, 256]}}
²
®regularization_losses
¯	variables
°trainable_variables
±	keras_api
¨__call__
+©&call_and_return_all_conditional_losses"
_tf_keras_layer{"class_name": "TensorFlowOpLayer", "name": "tf_op_layer_concat_6", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "concat_6", "trainable": true, "dtype": "float32", "node_def": {"name": "concat_6", "op": "ConcatV2", "input": ["upsample_6/BiasAdd", "re_lu_3/Relu", "concat_6/axis"], "attr": {"Tidx": {"type": "DT_INT32"}, "T": {"type": "DT_FLOAT"}, "N": {"i": "2"}}}, "constants": {"2": -1}}}
û	
²kernel
	³bias
´regularization_losses
µ	variables
¶trainable_variables
·	keras_api
ª__call__
+«&call_and_return_all_conditional_losses"Î
_tf_keras_layer´{"class_name": "Conv2D", "name": "conv_6", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv_6", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 256}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 12, 24, 256]}}
Ç	
	¸axis

¹gamma
	ºbeta
»moving_mean
¼moving_variance
½regularization_losses
¾	variables
¿trainable_variables
À	keras_api
¬__call__
+­&call_and_return_all_conditional_losses"è
_tf_keras_layerÎ{"class_name": "BatchNormalization", "name": "batch_normalization_5", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_5", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 12, 24, 128]}}
ñ
Áregularization_losses
Â	variables
Ãtrainable_variables
Ä	keras_api
®__call__
+¯&call_and_return_all_conditional_losses"Ü
_tf_keras_layerÂ{"class_name": "ReLU", "name": "re_lu_5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "re_lu_5", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}}
¤

Åkernel
	Æbias
Çregularization_losses
È	variables
Étrainable_variables
Ê	keras_api
°__call__
+±&call_and_return_all_conditional_losses"÷
_tf_keras_layerÝ{"class_name": "Conv2DTranspose", "name": "upsample_7", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "upsample_7", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [2, 2]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 12, 24, 128]}}
²
Ëregularization_losses
Ì	variables
Ítrainable_variables
Î	keras_api
²__call__
+³&call_and_return_all_conditional_losses"
_tf_keras_layer{"class_name": "TensorFlowOpLayer", "name": "tf_op_layer_concat_7", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "concat_7", "trainable": true, "dtype": "float32", "node_def": {"name": "concat_7", "op": "ConcatV2", "input": ["upsample_7/BiasAdd", "re_lu_2/Relu", "concat_7/axis"], "attr": {"Tidx": {"type": "DT_INT32"}, "T": {"type": "DT_FLOAT"}, "N": {"i": "2"}}}, "constants": {"2": -1}}}
ú	
Ïkernel
	Ðbias
Ñregularization_losses
Ò	variables
Ótrainable_variables
Ô	keras_api
´__call__
+µ&call_and_return_all_conditional_losses"Í
_tf_keras_layer³{"class_name": "Conv2D", "name": "conv_7", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv_7", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 24, 48, 128]}}
Å	
	Õaxis

Ögamma
	×beta
Ømoving_mean
Ùmoving_variance
Úregularization_losses
Û	variables
Ütrainable_variables
Ý	keras_api
¶__call__
+·&call_and_return_all_conditional_losses"æ
_tf_keras_layerÌ{"class_name": "BatchNormalization", "name": "batch_normalization_6", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_6", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 24, 48, 64]}}
ñ
Þregularization_losses
ß	variables
àtrainable_variables
á	keras_api
¸__call__
+¹&call_and_return_all_conditional_losses"Ü
_tf_keras_layerÂ{"class_name": "ReLU", "name": "re_lu_6", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "re_lu_6", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}}
¢

âkernel
	ãbias
äregularization_losses
å	variables
ætrainable_variables
ç	keras_api
º__call__
+»&call_and_return_all_conditional_losses"õ
_tf_keras_layerÛ{"class_name": "Conv2DTranspose", "name": "upsample_8", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "upsample_8", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [2, 2]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 24, 48, 64]}}
²
èregularization_losses
é	variables
êtrainable_variables
ë	keras_api
¼__call__
+½&call_and_return_all_conditional_losses"
_tf_keras_layer{"class_name": "TensorFlowOpLayer", "name": "tf_op_layer_concat_8", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "concat_8", "trainable": true, "dtype": "float32", "node_def": {"name": "concat_8", "op": "ConcatV2", "input": ["upsample_8/BiasAdd", "re_lu_1/Relu", "concat_8/axis"], "attr": {"Tidx": {"type": "DT_INT32"}, "T": {"type": "DT_FLOAT"}, "N": {"i": "2"}}}, "constants": {"2": -1}}}
ø	
ìkernel
	íbias
îregularization_losses
ï	variables
ðtrainable_variables
ñ	keras_api
¾__call__
+¿&call_and_return_all_conditional_losses"Ë
_tf_keras_layer±{"class_name": "Conv2D", "name": "conv_8", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv_8", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 48, 96, 64]}}
Å	
	òaxis

ógamma
	ôbeta
õmoving_mean
ömoving_variance
÷regularization_losses
ø	variables
ùtrainable_variables
ú	keras_api
À__call__
+Á&call_and_return_all_conditional_losses"æ
_tf_keras_layerÌ{"class_name": "BatchNormalization", "name": "batch_normalization_7", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_7", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 48, 96, 32]}}
ñ
ûregularization_losses
ü	variables
ýtrainable_variables
þ	keras_api
Â__call__
+Ã&call_and_return_all_conditional_losses"Ü
_tf_keras_layerÂ{"class_name": "ReLU", "name": "re_lu_7", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "re_lu_7", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}}
¢

ÿkernel
	bias
regularization_losses
	variables
trainable_variables
	keras_api
Ä__call__
+Å&call_and_return_all_conditional_losses"õ
_tf_keras_layerÛ{"class_name": "Conv2DTranspose", "name": "upsample_9", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "upsample_9", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [2, 2]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 48, 96, 32]}}
°
regularization_losses
	variables
trainable_variables
	keras_api
Æ__call__
+Ç&call_and_return_all_conditional_losses"
_tf_keras_layer{"class_name": "TensorFlowOpLayer", "name": "tf_op_layer_concat_9", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "concat_9", "trainable": true, "dtype": "float32", "node_def": {"name": "concat_9", "op": "ConcatV2", "input": ["upsample_9/BiasAdd", "re_lu/Relu", "concat_9/axis"], "attr": {"Tidx": {"type": "DT_INT32"}, "T": {"type": "DT_FLOAT"}, "N": {"i": "2"}}}, "constants": {"2": -1}}}
ù	
kernel
	bias
regularization_losses
	variables
trainable_variables
	keras_api
È__call__
+É&call_and_return_all_conditional_losses"Ì
_tf_keras_layer²{"class_name": "Conv2D", "name": "conv_9", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv_9", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 96, 192, 32]}}
Æ	
	axis

gamma
	beta
moving_mean
moving_variance
regularization_losses
	variables
trainable_variables
	keras_api
Ê__call__
+Ë&call_and_return_all_conditional_losses"ç
_tf_keras_layerÍ{"class_name": "BatchNormalization", "name": "batch_normalization_8", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_8", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 16}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 96, 192, 16]}}
ñ
regularization_losses
	variables
trainable_variables
	keras_api
Ì__call__
+Í&call_and_return_all_conditional_losses"Ü
_tf_keras_layerÂ{"class_name": "ReLU", "name": "re_lu_8", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "re_lu_8", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}}
ö	
kernel
	bias
regularization_losses
	variables
 trainable_variables
¡	keras_api
Î__call__
+Ï&call_and_return_all_conditional_losses"É
_tf_keras_layer¯{"class_name": "Conv2D", "name": "final", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "final", "trainable": true, "dtype": "float32", "filters": 2, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 16}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 96, 192, 16]}}
 "
trackable_list_wrapper
Â
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
¹34
º35
»36
¼37
Å38
Æ39
Ï40
Ð41
Ö42
×43
Ø44
Ù45
â46
ã47
ì48
í49
ó50
ô51
õ52
ö53
ÿ54
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
¹24
º25
Å26
Æ27
Ï28
Ð29
Ö30
×31
â32
ã33
ì34
í35
ó36
ô37
ÿ38
39
40
41
42
43
44
45"
trackable_list_wrapper
Ó
,regularization_losses
¢non_trainable_variables
-	variables
£metrics
 ¤layer_regularization_losses
¥layer_metrics
¦layers
.trainable_variables
ù__call__
û_default_save_signature
+ú&call_and_return_all_conditional_losses
'ú"call_and_return_conditional_losses"
_generic_user_object
-
Ðserving_default"
signature_map
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
1regularization_losses
§non_trainable_variables
2	variables
¨metrics
 ©layer_regularization_losses
ªlayer_metrics
«layers
3trainable_variables
ü__call__
+ý&call_and_return_all_conditional_losses
'ý"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
5regularization_losses
¬non_trainable_variables
6	variables
­metrics
 ®layer_regularization_losses
¯layer_metrics
°layers
7trainable_variables
þ__call__
+ÿ&call_and_return_all_conditional_losses
'ÿ"call_and_return_conditional_losses"
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
µ
;regularization_losses
±non_trainable_variables
<	variables
²metrics
 ³layer_regularization_losses
´layer_metrics
µlayers
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
µ
Dregularization_losses
¶non_trainable_variables
E	variables
·metrics
 ¸layer_regularization_losses
¹layer_metrics
ºlayers
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
µ
Hregularization_losses
»non_trainable_variables
I	variables
¼metrics
 ½layer_regularization_losses
¾layer_metrics
¿layers
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
µ
Lregularization_losses
Ànon_trainable_variables
M	variables
Ámetrics
 Âlayer_regularization_losses
Ãlayer_metrics
Älayers
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
µ
Rregularization_losses
Ånon_trainable_variables
S	variables
Æmetrics
 Çlayer_regularization_losses
Èlayer_metrics
Élayers
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
µ
[regularization_losses
Ênon_trainable_variables
\	variables
Ëmetrics
 Ìlayer_regularization_losses
Ílayer_metrics
Îlayers
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
µ
_regularization_losses
Ïnon_trainable_variables
`	variables
Ðmetrics
 Ñlayer_regularization_losses
Òlayer_metrics
Ólayers
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
µ
cregularization_losses
Ônon_trainable_variables
d	variables
Õmetrics
 Ölayer_regularization_losses
×layer_metrics
Ølayers
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
µ
iregularization_losses
Ùnon_trainable_variables
j	variables
Úmetrics
 Ûlayer_regularization_losses
Ülayer_metrics
Ýlayers
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
µ
rregularization_losses
Þnon_trainable_variables
s	variables
ßmetrics
 àlayer_regularization_losses
álayer_metrics
âlayers
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
µ
vregularization_losses
ãnon_trainable_variables
w	variables
ämetrics
 ålayer_regularization_losses
ælayer_metrics
çlayers
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
µ
zregularization_losses
ènon_trainable_variables
{	variables
émetrics
 êlayer_regularization_losses
ëlayer_metrics
ìlayers
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
¸
regularization_losses
ínon_trainable_variables
	variables
îmetrics
 ïlayer_regularization_losses
ðlayer_metrics
ñlayers
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
¸
regularization_losses
ònon_trainable_variables
	variables
ómetrics
 ôlayer_regularization_losses
õlayer_metrics
ölayers
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
¸
regularization_losses
÷non_trainable_variables
	variables
ømetrics
 ùlayer_regularization_losses
úlayer_metrics
ûlayers
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
¸
regularization_losses
ünon_trainable_variables
	variables
ýmetrics
 þlayer_regularization_losses
ÿlayer_metrics
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
¸
regularization_losses
non_trainable_variables
	variables
metrics
 layer_regularization_losses
layer_metrics
layers
trainable_variables
 __call__
+¡&call_and_return_all_conditional_losses
'¡"call_and_return_conditional_losses"
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
¸
 regularization_losses
non_trainable_variables
¡	variables
metrics
 layer_regularization_losses
layer_metrics
layers
¢trainable_variables
¢__call__
+£&call_and_return_all_conditional_losses
'£"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
¤regularization_losses
non_trainable_variables
¥	variables
metrics
 layer_regularization_losses
layer_metrics
layers
¦trainable_variables
¤__call__
+¥&call_and_return_all_conditional_losses
'¥"call_and_return_conditional_losses"
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
¸
ªregularization_losses
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
¸
®regularization_losses
non_trainable_variables
¯	variables
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
¸
´regularization_losses
non_trainable_variables
µ	variables
metrics
 layer_regularization_losses
layer_metrics
layers
¶trainable_variables
ª__call__
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
¹0
º1
»2
¼3"
trackable_list_wrapper
0
¹0
º1"
trackable_list_wrapper
¸
½regularization_losses
non_trainable_variables
¾	variables
 metrics
 ¡layer_regularization_losses
¢layer_metrics
£layers
¿trainable_variables
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
¸
Áregularization_losses
¤non_trainable_variables
Â	variables
¥metrics
 ¦layer_regularization_losses
§layer_metrics
¨layers
Ãtrainable_variables
®__call__
+¯&call_and_return_all_conditional_losses
'¯"call_and_return_conditional_losses"
_generic_user_object
,:*@2upsample_7/kernel
:@2upsample_7/bias
 "
trackable_list_wrapper
0
Å0
Æ1"
trackable_list_wrapper
0
Å0
Æ1"
trackable_list_wrapper
¸
Çregularization_losses
©non_trainable_variables
È	variables
ªmetrics
 «layer_regularization_losses
¬layer_metrics
­layers
Étrainable_variables
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
¸
Ëregularization_losses
®non_trainable_variables
Ì	variables
¯metrics
 °layer_regularization_losses
±layer_metrics
²layers
Ítrainable_variables
²__call__
+³&call_and_return_all_conditional_losses
'³"call_and_return_conditional_losses"
_generic_user_object
(:&@2conv_7/kernel
:@2conv_7/bias
 "
trackable_list_wrapper
0
Ï0
Ð1"
trackable_list_wrapper
0
Ï0
Ð1"
trackable_list_wrapper
¸
Ñregularization_losses
³non_trainable_variables
Ò	variables
´metrics
 µlayer_regularization_losses
¶layer_metrics
·layers
Ótrainable_variables
´__call__
+µ&call_and_return_all_conditional_losses
'µ"call_and_return_conditional_losses"
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
Ö0
×1
Ø2
Ù3"
trackable_list_wrapper
0
Ö0
×1"
trackable_list_wrapper
¸
Úregularization_losses
¸non_trainable_variables
Û	variables
¹metrics
 ºlayer_regularization_losses
»layer_metrics
¼layers
Ütrainable_variables
¶__call__
+·&call_and_return_all_conditional_losses
'·"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
Þregularization_losses
½non_trainable_variables
ß	variables
¾metrics
 ¿layer_regularization_losses
Àlayer_metrics
Álayers
àtrainable_variables
¸__call__
+¹&call_and_return_all_conditional_losses
'¹"call_and_return_conditional_losses"
_generic_user_object
+:) @2upsample_8/kernel
: 2upsample_8/bias
 "
trackable_list_wrapper
0
â0
ã1"
trackable_list_wrapper
0
â0
ã1"
trackable_list_wrapper
¸
äregularization_losses
Ânon_trainable_variables
å	variables
Ãmetrics
 Älayer_regularization_losses
Ålayer_metrics
Ælayers
ætrainable_variables
º__call__
+»&call_and_return_all_conditional_losses
'»"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
èregularization_losses
Çnon_trainable_variables
é	variables
Èmetrics
 Élayer_regularization_losses
Êlayer_metrics
Ëlayers
êtrainable_variables
¼__call__
+½&call_and_return_all_conditional_losses
'½"call_and_return_conditional_losses"
_generic_user_object
':%@ 2conv_8/kernel
: 2conv_8/bias
 "
trackable_list_wrapper
0
ì0
í1"
trackable_list_wrapper
0
ì0
í1"
trackable_list_wrapper
¸
îregularization_losses
Ìnon_trainable_variables
ï	variables
Ímetrics
 Îlayer_regularization_losses
Ïlayer_metrics
Ðlayers
ðtrainable_variables
¾__call__
+¿&call_and_return_all_conditional_losses
'¿"call_and_return_conditional_losses"
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
ó0
ô1
õ2
ö3"
trackable_list_wrapper
0
ó0
ô1"
trackable_list_wrapper
¸
÷regularization_losses
Ñnon_trainable_variables
ø	variables
Òmetrics
 Ólayer_regularization_losses
Ôlayer_metrics
Õlayers
ùtrainable_variables
À__call__
+Á&call_and_return_all_conditional_losses
'Á"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
ûregularization_losses
Önon_trainable_variables
ü	variables
×metrics
 Ølayer_regularization_losses
Ùlayer_metrics
Úlayers
ýtrainable_variables
Â__call__
+Ã&call_and_return_all_conditional_losses
'Ã"call_and_return_conditional_losses"
_generic_user_object
+:) 2upsample_9/kernel
:2upsample_9/bias
 "
trackable_list_wrapper
0
ÿ0
1"
trackable_list_wrapper
0
ÿ0
1"
trackable_list_wrapper
¸
regularization_losses
Ûnon_trainable_variables
	variables
Ümetrics
 Ýlayer_regularization_losses
Þlayer_metrics
ßlayers
trainable_variables
Ä__call__
+Å&call_and_return_all_conditional_losses
'Å"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
regularization_losses
ànon_trainable_variables
	variables
ámetrics
 âlayer_regularization_losses
ãlayer_metrics
älayers
trainable_variables
Æ__call__
+Ç&call_and_return_all_conditional_losses
'Ç"call_and_return_conditional_losses"
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
¸
regularization_losses
ånon_trainable_variables
	variables
æmetrics
 çlayer_regularization_losses
èlayer_metrics
élayers
trainable_variables
È__call__
+É&call_and_return_all_conditional_losses
'É"call_and_return_conditional_losses"
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
¸
regularization_losses
ênon_trainable_variables
	variables
ëmetrics
 ìlayer_regularization_losses
ílayer_metrics
îlayers
trainable_variables
Ê__call__
+Ë&call_and_return_all_conditional_losses
'Ë"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
regularization_losses
ïnon_trainable_variables
	variables
ðmetrics
 ñlayer_regularization_losses
òlayer_metrics
ólayers
trainable_variables
Ì__call__
+Í&call_and_return_all_conditional_losses
'Í"call_and_return_conditional_losses"
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
¸
regularization_losses
ônon_trainable_variables
	variables
õmetrics
 ölayer_regularization_losses
÷layer_metrics
ølayers
 trainable_variables
Î__call__
+Ï&call_and_return_all_conditional_losses
'Ï"call_and_return_conditional_losses"
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
¼11
Ø12
Ù13
õ14
ö15
16
17"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
î
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
¼1"
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
Ø0
Ù1"
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
õ0
ö1"
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
2
0__inference_functional_1_layer_call_fn_107946919
0__inference_functional_1_layer_call_fn_107947787
0__inference_functional_1_layer_call_fn_107946612
0__inference_functional_1_layer_call_fn_107947920À
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
kwonlydefaultsª 
annotationsª *
 
ú2÷
K__inference_functional_1_layer_call_and_return_conditional_losses_107947654
K__inference_functional_1_layer_call_and_return_conditional_losses_107946304
K__inference_functional_1_layer_call_and_return_conditional_losses_107947363
K__inference_functional_1_layer_call_and_return_conditional_losses_107946130À
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
kwonlydefaultsª 
annotationsª *
 
î2ë
$__inference__wrapped_model_107943799Â
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
annotationsª *2¢/
-*

imageInputÿÿÿÿÿÿÿÿÿ`À
á2Þ
7__inference_tf_op_layer_RealDiv_layer_call_fn_107947931¢
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
annotationsª *
 
ü2ù
R__inference_tf_op_layer_RealDiv_layer_call_and_return_conditional_losses_107947926¢
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
annotationsª *
 
Ý2Ú
3__inference_tf_op_layer_Sub_layer_call_fn_107947942¢
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
annotationsª *
 
ø2õ
N__inference_tf_op_layer_Sub_layer_call_and_return_conditional_losses_107947937¢
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
annotationsª *
 
Ô2Ñ
*__inference_conv_1_layer_call_fn_107947961¢
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
annotationsª *
 
ï2ì
E__inference_conv_1_layer_call_and_return_conditional_losses_107947952¢
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
annotationsª *
 
2
7__inference_batch_normalization_layer_call_fn_107948076
7__inference_batch_normalization_layer_call_fn_107948025
7__inference_batch_normalization_layer_call_fn_107948089
7__inference_batch_normalization_layer_call_fn_107948012´
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
kwonlydefaultsª 
annotationsª *
 
2
R__inference_batch_normalization_layer_call_and_return_conditional_losses_107947981
R__inference_batch_normalization_layer_call_and_return_conditional_losses_107947999
R__inference_batch_normalization_layer_call_and_return_conditional_losses_107948045
R__inference_batch_normalization_layer_call_and_return_conditional_losses_107948063´
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
kwonlydefaultsª 
annotationsª *
 
Ó2Ð
)__inference_re_lu_layer_call_fn_107948099¢
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
annotationsª *
 
î2ë
D__inference_re_lu_layer_call_and_return_conditional_losses_107948094¢
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
annotationsª *
 
2
1__inference_max_pooling2d_layer_call_fn_107943915à
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
annotationsª *@¢=
;84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
´2±
L__inference_max_pooling2d_layer_call_and_return_conditional_losses_107943909à
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
annotationsª *@¢=
;84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
Ô2Ñ
*__inference_conv_2_layer_call_fn_107948118¢
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
annotationsª *
 
ï2ì
E__inference_conv_2_layer_call_and_return_conditional_losses_107948109¢
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
annotationsª *
 
¦2£
9__inference_batch_normalization_1_layer_call_fn_107948182
9__inference_batch_normalization_1_layer_call_fn_107948169
9__inference_batch_normalization_1_layer_call_fn_107948233
9__inference_batch_normalization_1_layer_call_fn_107948246´
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
kwonlydefaultsª 
annotationsª *
 
2
T__inference_batch_normalization_1_layer_call_and_return_conditional_losses_107948220
T__inference_batch_normalization_1_layer_call_and_return_conditional_losses_107948138
T__inference_batch_normalization_1_layer_call_and_return_conditional_losses_107948156
T__inference_batch_normalization_1_layer_call_and_return_conditional_losses_107948202´
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
kwonlydefaultsª 
annotationsª *
 
Õ2Ò
+__inference_re_lu_1_layer_call_fn_107948256¢
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
annotationsª *
 
ð2í
F__inference_re_lu_1_layer_call_and_return_conditional_losses_107948251¢
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
annotationsª *
 
2
3__inference_max_pooling2d_1_layer_call_fn_107944031à
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
annotationsª *@¢=
;84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
¶2³
N__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_107944025à
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
annotationsª *@¢=
;84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
Ô2Ñ
*__inference_conv_3_layer_call_fn_107948275¢
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
annotationsª *
 
ï2ì
E__inference_conv_3_layer_call_and_return_conditional_losses_107948266¢
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
annotationsª *
 
¦2£
9__inference_batch_normalization_2_layer_call_fn_107948326
9__inference_batch_normalization_2_layer_call_fn_107948390
9__inference_batch_normalization_2_layer_call_fn_107948339
9__inference_batch_normalization_2_layer_call_fn_107948403´
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
kwonlydefaultsª 
annotationsª *
 
2
T__inference_batch_normalization_2_layer_call_and_return_conditional_losses_107948313
T__inference_batch_normalization_2_layer_call_and_return_conditional_losses_107948295
T__inference_batch_normalization_2_layer_call_and_return_conditional_losses_107948377
T__inference_batch_normalization_2_layer_call_and_return_conditional_losses_107948359´
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
kwonlydefaultsª 
annotationsª *
 
Õ2Ò
+__inference_re_lu_2_layer_call_fn_107948413¢
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
annotationsª *
 
ð2í
F__inference_re_lu_2_layer_call_and_return_conditional_losses_107948408¢
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
annotationsª *
 
2
3__inference_max_pooling2d_2_layer_call_fn_107944147à
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
annotationsª *@¢=
;84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
¶2³
N__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_107944141à
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
annotationsª *@¢=
;84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
Ô2Ñ
*__inference_conv_4_layer_call_fn_107948432¢
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
annotationsª *
 
ï2ì
E__inference_conv_4_layer_call_and_return_conditional_losses_107948423¢
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
annotationsª *
 
¦2£
9__inference_batch_normalization_3_layer_call_fn_107948560
9__inference_batch_normalization_3_layer_call_fn_107948483
9__inference_batch_normalization_3_layer_call_fn_107948496
9__inference_batch_normalization_3_layer_call_fn_107948547´
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
kwonlydefaultsª 
annotationsª *
 
2
T__inference_batch_normalization_3_layer_call_and_return_conditional_losses_107948534
T__inference_batch_normalization_3_layer_call_and_return_conditional_losses_107948470
T__inference_batch_normalization_3_layer_call_and_return_conditional_losses_107948452
T__inference_batch_normalization_3_layer_call_and_return_conditional_losses_107948516´
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
kwonlydefaultsª 
annotationsª *
 
Õ2Ò
+__inference_re_lu_3_layer_call_fn_107948570¢
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
annotationsª *
 
ð2í
F__inference_re_lu_3_layer_call_and_return_conditional_losses_107948565¢
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
annotationsª *
 
2
3__inference_max_pooling2d_3_layer_call_fn_107944263à
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
annotationsª *@¢=
;84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
¶2³
N__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_107944257à
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
annotationsª *@¢=
;84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
Ô2Ñ
*__inference_conv_5_layer_call_fn_107948589¢
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
annotationsª *
 
ï2ì
E__inference_conv_5_layer_call_and_return_conditional_losses_107948580¢
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
annotationsª *
 
¦2£
9__inference_batch_normalization_4_layer_call_fn_107948640
9__inference_batch_normalization_4_layer_call_fn_107948704
9__inference_batch_normalization_4_layer_call_fn_107948653
9__inference_batch_normalization_4_layer_call_fn_107948717´
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
kwonlydefaultsª 
annotationsª *
 
2
T__inference_batch_normalization_4_layer_call_and_return_conditional_losses_107948627
T__inference_batch_normalization_4_layer_call_and_return_conditional_losses_107948609
T__inference_batch_normalization_4_layer_call_and_return_conditional_losses_107948673
T__inference_batch_normalization_4_layer_call_and_return_conditional_losses_107948691´
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
kwonlydefaultsª 
annotationsª *
 
Õ2Ò
+__inference_re_lu_4_layer_call_fn_107948727¢
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
annotationsª *
 
ð2í
F__inference_re_lu_4_layer_call_and_return_conditional_losses_107948722¢
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
annotationsª *
 
2
.__inference_upsample_6_layer_call_fn_107944415Ø
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
annotationsª *8¢5
30,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
©2¦
I__inference_upsample_6_layer_call_and_return_conditional_losses_107944405Ø
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
annotationsª *8¢5
30,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
â2ß
8__inference_tf_op_layer_concat_6_layer_call_fn_107948740¢
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
annotationsª *
 
ý2ú
S__inference_tf_op_layer_concat_6_layer_call_and_return_conditional_losses_107948734¢
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
annotationsª *
 
Ô2Ñ
*__inference_conv_6_layer_call_fn_107948759¢
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
annotationsª *
 
ï2ì
E__inference_conv_6_layer_call_and_return_conditional_losses_107948750¢
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
annotationsª *
 
¦2£
9__inference_batch_normalization_5_layer_call_fn_107948887
9__inference_batch_normalization_5_layer_call_fn_107948810
9__inference_batch_normalization_5_layer_call_fn_107948823
9__inference_batch_normalization_5_layer_call_fn_107948874´
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
kwonlydefaultsª 
annotationsª *
 
2
T__inference_batch_normalization_5_layer_call_and_return_conditional_losses_107948797
T__inference_batch_normalization_5_layer_call_and_return_conditional_losses_107948779
T__inference_batch_normalization_5_layer_call_and_return_conditional_losses_107948861
T__inference_batch_normalization_5_layer_call_and_return_conditional_losses_107948843´
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
kwonlydefaultsª 
annotationsª *
 
Õ2Ò
+__inference_re_lu_5_layer_call_fn_107948897¢
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
annotationsª *
 
ð2í
F__inference_re_lu_5_layer_call_and_return_conditional_losses_107948892¢
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
annotationsª *
 
2
.__inference_upsample_7_layer_call_fn_107944567Ø
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
annotationsª *8¢5
30,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
©2¦
I__inference_upsample_7_layer_call_and_return_conditional_losses_107944557Ø
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
annotationsª *8¢5
30,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
â2ß
8__inference_tf_op_layer_concat_7_layer_call_fn_107948910¢
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
annotationsª *
 
ý2ú
S__inference_tf_op_layer_concat_7_layer_call_and_return_conditional_losses_107948904¢
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
annotationsª *
 
Ô2Ñ
*__inference_conv_7_layer_call_fn_107948929¢
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
annotationsª *
 
ï2ì
E__inference_conv_7_layer_call_and_return_conditional_losses_107948920¢
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
annotationsª *
 
¦2£
9__inference_batch_normalization_6_layer_call_fn_107948980
9__inference_batch_normalization_6_layer_call_fn_107949057
9__inference_batch_normalization_6_layer_call_fn_107948993
9__inference_batch_normalization_6_layer_call_fn_107949044´
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
kwonlydefaultsª 
annotationsª *
 
2
T__inference_batch_normalization_6_layer_call_and_return_conditional_losses_107948967
T__inference_batch_normalization_6_layer_call_and_return_conditional_losses_107948949
T__inference_batch_normalization_6_layer_call_and_return_conditional_losses_107949031
T__inference_batch_normalization_6_layer_call_and_return_conditional_losses_107949013´
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
kwonlydefaultsª 
annotationsª *
 
Õ2Ò
+__inference_re_lu_6_layer_call_fn_107949067¢
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
annotationsª *
 
ð2í
F__inference_re_lu_6_layer_call_and_return_conditional_losses_107949062¢
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
annotationsª *
 
2
.__inference_upsample_8_layer_call_fn_107944719×
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
annotationsª *7¢4
2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
¨2¥
I__inference_upsample_8_layer_call_and_return_conditional_losses_107944709×
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
annotationsª *7¢4
2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
â2ß
8__inference_tf_op_layer_concat_8_layer_call_fn_107949080¢
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
annotationsª *
 
ý2ú
S__inference_tf_op_layer_concat_8_layer_call_and_return_conditional_losses_107949074¢
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
annotationsª *
 
Ô2Ñ
*__inference_conv_8_layer_call_fn_107949099¢
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
annotationsª *
 
ï2ì
E__inference_conv_8_layer_call_and_return_conditional_losses_107949090¢
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
annotationsª *
 
¦2£
9__inference_batch_normalization_7_layer_call_fn_107949214
9__inference_batch_normalization_7_layer_call_fn_107949150
9__inference_batch_normalization_7_layer_call_fn_107949227
9__inference_batch_normalization_7_layer_call_fn_107949163´
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
kwonlydefaultsª 
annotationsª *
 
2
T__inference_batch_normalization_7_layer_call_and_return_conditional_losses_107949119
T__inference_batch_normalization_7_layer_call_and_return_conditional_losses_107949183
T__inference_batch_normalization_7_layer_call_and_return_conditional_losses_107949137
T__inference_batch_normalization_7_layer_call_and_return_conditional_losses_107949201´
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
kwonlydefaultsª 
annotationsª *
 
Õ2Ò
+__inference_re_lu_7_layer_call_fn_107949237¢
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
annotationsª *
 
ð2í
F__inference_re_lu_7_layer_call_and_return_conditional_losses_107949232¢
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
annotationsª *
 
2
.__inference_upsample_9_layer_call_fn_107944871×
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
annotationsª *7¢4
2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
¨2¥
I__inference_upsample_9_layer_call_and_return_conditional_losses_107944861×
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
annotationsª *7¢4
2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
â2ß
8__inference_tf_op_layer_concat_9_layer_call_fn_107949250¢
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
annotationsª *
 
ý2ú
S__inference_tf_op_layer_concat_9_layer_call_and_return_conditional_losses_107949244¢
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
annotationsª *
 
Ô2Ñ
*__inference_conv_9_layer_call_fn_107949269¢
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
annotationsª *
 
ï2ì
E__inference_conv_9_layer_call_and_return_conditional_losses_107949260¢
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
annotationsª *
 
¦2£
9__inference_batch_normalization_8_layer_call_fn_107949320
9__inference_batch_normalization_8_layer_call_fn_107949397
9__inference_batch_normalization_8_layer_call_fn_107949333
9__inference_batch_normalization_8_layer_call_fn_107949384´
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
kwonlydefaultsª 
annotationsª *
 
2
T__inference_batch_normalization_8_layer_call_and_return_conditional_losses_107949307
T__inference_batch_normalization_8_layer_call_and_return_conditional_losses_107949371
T__inference_batch_normalization_8_layer_call_and_return_conditional_losses_107949353
T__inference_batch_normalization_8_layer_call_and_return_conditional_losses_107949289´
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
kwonlydefaultsª 
annotationsª *
 
Õ2Ò
+__inference_re_lu_8_layer_call_fn_107949407¢
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
annotationsª *
 
ð2í
F__inference_re_lu_8_layer_call_and_return_conditional_losses_107949402¢
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
annotationsª *
 
Ó2Ð
)__inference_final_layer_call_fn_107949426¢
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
annotationsª *
 
î2ë
D__inference_final_layer_call_and_return_conditional_losses_107949417¢
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
annotationsª *
 
9B7
'__inference_signature_wrapper_107947054
imageInput
$__inference__wrapped_model_107943799äl9:@ABCPQWXYZghnopq~¨©²³¹º»¼ÅÆÏÐÖ×ØÙâãìíóôõöÿ<¢9
2¢/
-*

imageInputÿÿÿÿÿÿÿÿÿ`À
ª "6ª3
1
final(%
finalÿÿÿÿÿÿÿÿÿ`Àï
T__inference_batch_normalization_1_layer_call_and_return_conditional_losses_107948138WXYZM¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
p
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 ï
T__inference_batch_normalization_1_layer_call_and_return_conditional_losses_107948156WXYZM¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
p 
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 Ê
T__inference_batch_normalization_1_layer_call_and_return_conditional_losses_107948202rWXYZ;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ0` 
p
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ0` 
 Ê
T__inference_batch_normalization_1_layer_call_and_return_conditional_losses_107948220rWXYZ;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ0` 
p 
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ0` 
 Ç
9__inference_batch_normalization_1_layer_call_fn_107948169WXYZM¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
p
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ Ç
9__inference_batch_normalization_1_layer_call_fn_107948182WXYZM¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
p 
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ ¢
9__inference_batch_normalization_1_layer_call_fn_107948233eWXYZ;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ0` 
p
ª " ÿÿÿÿÿÿÿÿÿ0` ¢
9__inference_batch_normalization_1_layer_call_fn_107948246eWXYZ;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ0` 
p 
ª " ÿÿÿÿÿÿÿÿÿ0` ï
T__inference_batch_normalization_2_layer_call_and_return_conditional_losses_107948295nopqM¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
p
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 ï
T__inference_batch_normalization_2_layer_call_and_return_conditional_losses_107948313nopqM¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
p 
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 Ê
T__inference_batch_normalization_2_layer_call_and_return_conditional_losses_107948359rnopq;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ0@
p
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ0@
 Ê
T__inference_batch_normalization_2_layer_call_and_return_conditional_losses_107948377rnopq;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ0@
p 
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ0@
 Ç
9__inference_batch_normalization_2_layer_call_fn_107948326nopqM¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
p
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@Ç
9__inference_batch_normalization_2_layer_call_fn_107948339nopqM¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
p 
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@¢
9__inference_batch_normalization_2_layer_call_fn_107948390enopq;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ0@
p
ª " ÿÿÿÿÿÿÿÿÿ0@¢
9__inference_batch_normalization_2_layer_call_fn_107948403enopq;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ0@
p 
ª " ÿÿÿÿÿÿÿÿÿ0@õ
T__inference_batch_normalization_3_layer_call_and_return_conditional_losses_107948452N¢K
D¢A
;8
inputs,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p
ª "@¢=
63
0,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 õ
T__inference_batch_normalization_3_layer_call_and_return_conditional_losses_107948470N¢K
D¢A
;8
inputs,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 
ª "@¢=
63
0,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Ð
T__inference_batch_normalization_3_layer_call_and_return_conditional_losses_107948516x<¢9
2¢/
)&
inputsÿÿÿÿÿÿÿÿÿ
p
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿ
 Ð
T__inference_batch_normalization_3_layer_call_and_return_conditional_losses_107948534x<¢9
2¢/
)&
inputsÿÿÿÿÿÿÿÿÿ
p 
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿ
 Í
9__inference_batch_normalization_3_layer_call_fn_107948483N¢K
D¢A
;8
inputs,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p
ª "30,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÍ
9__inference_batch_normalization_3_layer_call_fn_107948496N¢K
D¢A
;8
inputs,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 
ª "30,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¨
9__inference_batch_normalization_3_layer_call_fn_107948547k<¢9
2¢/
)&
inputsÿÿÿÿÿÿÿÿÿ
p
ª "!ÿÿÿÿÿÿÿÿÿ¨
9__inference_batch_normalization_3_layer_call_fn_107948560k<¢9
2¢/
)&
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "!ÿÿÿÿÿÿÿÿÿõ
T__inference_batch_normalization_4_layer_call_and_return_conditional_losses_107948609N¢K
D¢A
;8
inputs,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p
ª "@¢=
63
0,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 õ
T__inference_batch_normalization_4_layer_call_and_return_conditional_losses_107948627N¢K
D¢A
;8
inputs,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 
ª "@¢=
63
0,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Ð
T__inference_batch_normalization_4_layer_call_and_return_conditional_losses_107948673x<¢9
2¢/
)&
inputsÿÿÿÿÿÿÿÿÿ
p
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿ
 Ð
T__inference_batch_normalization_4_layer_call_and_return_conditional_losses_107948691x<¢9
2¢/
)&
inputsÿÿÿÿÿÿÿÿÿ
p 
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿ
 Í
9__inference_batch_normalization_4_layer_call_fn_107948640N¢K
D¢A
;8
inputs,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p
ª "30,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÍ
9__inference_batch_normalization_4_layer_call_fn_107948653N¢K
D¢A
;8
inputs,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 
ª "30,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¨
9__inference_batch_normalization_4_layer_call_fn_107948704k<¢9
2¢/
)&
inputsÿÿÿÿÿÿÿÿÿ
p
ª "!ÿÿÿÿÿÿÿÿÿ¨
9__inference_batch_normalization_4_layer_call_fn_107948717k<¢9
2¢/
)&
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "!ÿÿÿÿÿÿÿÿÿõ
T__inference_batch_normalization_5_layer_call_and_return_conditional_losses_107948779¹º»¼N¢K
D¢A
;8
inputs,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p
ª "@¢=
63
0,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 õ
T__inference_batch_normalization_5_layer_call_and_return_conditional_losses_107948797¹º»¼N¢K
D¢A
;8
inputs,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 
ª "@¢=
63
0,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Ð
T__inference_batch_normalization_5_layer_call_and_return_conditional_losses_107948843x¹º»¼<¢9
2¢/
)&
inputsÿÿÿÿÿÿÿÿÿ
p
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿ
 Ð
T__inference_batch_normalization_5_layer_call_and_return_conditional_losses_107948861x¹º»¼<¢9
2¢/
)&
inputsÿÿÿÿÿÿÿÿÿ
p 
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿ
 Í
9__inference_batch_normalization_5_layer_call_fn_107948810¹º»¼N¢K
D¢A
;8
inputs,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p
ª "30,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÍ
9__inference_batch_normalization_5_layer_call_fn_107948823¹º»¼N¢K
D¢A
;8
inputs,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 
ª "30,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¨
9__inference_batch_normalization_5_layer_call_fn_107948874k¹º»¼<¢9
2¢/
)&
inputsÿÿÿÿÿÿÿÿÿ
p
ª "!ÿÿÿÿÿÿÿÿÿ¨
9__inference_batch_normalization_5_layer_call_fn_107948887k¹º»¼<¢9
2¢/
)&
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "!ÿÿÿÿÿÿÿÿÿó
T__inference_batch_normalization_6_layer_call_and_return_conditional_losses_107948949Ö×ØÙM¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
p
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 ó
T__inference_batch_normalization_6_layer_call_and_return_conditional_losses_107948967Ö×ØÙM¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
p 
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 Î
T__inference_batch_normalization_6_layer_call_and_return_conditional_losses_107949013vÖ×ØÙ;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ0@
p
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ0@
 Î
T__inference_batch_normalization_6_layer_call_and_return_conditional_losses_107949031vÖ×ØÙ;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ0@
p 
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ0@
 Ë
9__inference_batch_normalization_6_layer_call_fn_107948980Ö×ØÙM¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
p
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@Ë
9__inference_batch_normalization_6_layer_call_fn_107948993Ö×ØÙM¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
p 
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@¦
9__inference_batch_normalization_6_layer_call_fn_107949044iÖ×ØÙ;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ0@
p
ª " ÿÿÿÿÿÿÿÿÿ0@¦
9__inference_batch_normalization_6_layer_call_fn_107949057iÖ×ØÙ;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ0@
p 
ª " ÿÿÿÿÿÿÿÿÿ0@ó
T__inference_batch_normalization_7_layer_call_and_return_conditional_losses_107949119óôõöM¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
p
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 ó
T__inference_batch_normalization_7_layer_call_and_return_conditional_losses_107949137óôõöM¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
p 
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 Î
T__inference_batch_normalization_7_layer_call_and_return_conditional_losses_107949183vóôõö;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ0` 
p
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ0` 
 Î
T__inference_batch_normalization_7_layer_call_and_return_conditional_losses_107949201vóôõö;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ0` 
p 
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ0` 
 Ë
9__inference_batch_normalization_7_layer_call_fn_107949150óôõöM¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
p
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ Ë
9__inference_batch_normalization_7_layer_call_fn_107949163óôõöM¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
p 
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ ¦
9__inference_batch_normalization_7_layer_call_fn_107949214ióôõö;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ0` 
p
ª " ÿÿÿÿÿÿÿÿÿ0` ¦
9__inference_batch_normalization_7_layer_call_fn_107949227ióôõö;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ0` 
p 
ª " ÿÿÿÿÿÿÿÿÿ0` ó
T__inference_batch_normalization_8_layer_call_and_return_conditional_losses_107949289M¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 ó
T__inference_batch_normalization_8_layer_call_and_return_conditional_losses_107949307M¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Ð
T__inference_batch_normalization_8_layer_call_and_return_conditional_losses_107949353x<¢9
2¢/
)&
inputsÿÿÿÿÿÿÿÿÿ`À
p
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿ`À
 Ð
T__inference_batch_normalization_8_layer_call_and_return_conditional_losses_107949371x<¢9
2¢/
)&
inputsÿÿÿÿÿÿÿÿÿ`À
p 
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿ`À
 Ë
9__inference_batch_normalization_8_layer_call_fn_107949320M¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿË
9__inference_batch_normalization_8_layer_call_fn_107949333M¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¨
9__inference_batch_normalization_8_layer_call_fn_107949384k<¢9
2¢/
)&
inputsÿÿÿÿÿÿÿÿÿ`À
p
ª "!ÿÿÿÿÿÿÿÿÿ`À¨
9__inference_batch_normalization_8_layer_call_fn_107949397k<¢9
2¢/
)&
inputsÿÿÿÿÿÿÿÿÿ`À
p 
ª "!ÿÿÿÿÿÿÿÿÿ`Àí
R__inference_batch_normalization_layer_call_and_return_conditional_losses_107947981@ABCM¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 í
R__inference_batch_normalization_layer_call_and_return_conditional_losses_107947999@ABCM¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Ê
R__inference_batch_normalization_layer_call_and_return_conditional_losses_107948045t@ABC<¢9
2¢/
)&
inputsÿÿÿÿÿÿÿÿÿ`À
p
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿ`À
 Ê
R__inference_batch_normalization_layer_call_and_return_conditional_losses_107948063t@ABC<¢9
2¢/
)&
inputsÿÿÿÿÿÿÿÿÿ`À
p 
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿ`À
 Å
7__inference_batch_normalization_layer_call_fn_107948012@ABCM¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÅ
7__inference_batch_normalization_layer_call_fn_107948025@ABCM¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¢
7__inference_batch_normalization_layer_call_fn_107948076g@ABC<¢9
2¢/
)&
inputsÿÿÿÿÿÿÿÿÿ`À
p
ª "!ÿÿÿÿÿÿÿÿÿ`À¢
7__inference_batch_normalization_layer_call_fn_107948089g@ABC<¢9
2¢/
)&
inputsÿÿÿÿÿÿÿÿÿ`À
p 
ª "!ÿÿÿÿÿÿÿÿÿ`À·
E__inference_conv_1_layer_call_and_return_conditional_losses_107947952n9:8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ`À
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿ`À
 
*__inference_conv_1_layer_call_fn_107947961a9:8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ`À
ª "!ÿÿÿÿÿÿÿÿÿ`Àµ
E__inference_conv_2_layer_call_and_return_conditional_losses_107948109lPQ7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ0`
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ0` 
 
*__inference_conv_2_layer_call_fn_107948118_PQ7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ0`
ª " ÿÿÿÿÿÿÿÿÿ0` µ
E__inference_conv_3_layer_call_and_return_conditional_losses_107948266lgh7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ0 
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ0@
 
*__inference_conv_3_layer_call_fn_107948275_gh7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ0 
ª " ÿÿÿÿÿÿÿÿÿ0@¶
E__inference_conv_4_layer_call_and_return_conditional_losses_107948423m~7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ@
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿ
 
*__inference_conv_4_layer_call_fn_107948432`~7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ@
ª "!ÿÿÿÿÿÿÿÿÿ¹
E__inference_conv_5_layer_call_and_return_conditional_losses_107948580p8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿ
 
*__inference_conv_5_layer_call_fn_107948589c8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ
ª "!ÿÿÿÿÿÿÿÿÿ¹
E__inference_conv_6_layer_call_and_return_conditional_losses_107948750p²³8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿ
 
*__inference_conv_6_layer_call_fn_107948759c²³8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ
ª "!ÿÿÿÿÿÿÿÿÿ¸
E__inference_conv_7_layer_call_and_return_conditional_losses_107948920oÏÐ8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ0
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ0@
 
*__inference_conv_7_layer_call_fn_107948929bÏÐ8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ0
ª " ÿÿÿÿÿÿÿÿÿ0@·
E__inference_conv_8_layer_call_and_return_conditional_losses_107949090nìí7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ0`@
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ0` 
 
*__inference_conv_8_layer_call_fn_107949099aìí7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ0`@
ª " ÿÿÿÿÿÿÿÿÿ0` ¹
E__inference_conv_9_layer_call_and_return_conditional_losses_107949260p8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ`À 
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿ`À
 
*__inference_conv_9_layer_call_fn_107949269c8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ`À 
ª "!ÿÿÿÿÿÿÿÿÿ`À¸
D__inference_final_layer_call_and_return_conditional_losses_107949417p8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ`À
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿ`À
 
)__inference_final_layer_call_fn_107949426c8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ`À
ª "!ÿÿÿÿÿÿÿÿÿ`À´
K__inference_functional_1_layer_call_and_return_conditional_losses_107946130äl9:@ABCPQWXYZghnopq~¨©²³¹º»¼ÅÆÏÐÖ×ØÙâãìíóôõöÿD¢A
:¢7
-*

imageInputÿÿÿÿÿÿÿÿÿ`À
p

 
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿ`À
 ´
K__inference_functional_1_layer_call_and_return_conditional_losses_107946304äl9:@ABCPQWXYZghnopq~¨©²³¹º»¼ÅÆÏÐÖ×ØÙâãìíóôõöÿD¢A
:¢7
-*

imageInputÿÿÿÿÿÿÿÿÿ`À
p 

 
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿ`À
 °
K__inference_functional_1_layer_call_and_return_conditional_losses_107947363àl9:@ABCPQWXYZghnopq~¨©²³¹º»¼ÅÆÏÐÖ×ØÙâãìíóôõöÿ@¢=
6¢3
)&
inputsÿÿÿÿÿÿÿÿÿ`À
p

 
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿ`À
 °
K__inference_functional_1_layer_call_and_return_conditional_losses_107947654àl9:@ABCPQWXYZghnopq~¨©²³¹º»¼ÅÆÏÐÖ×ØÙâãìíóôõöÿ@¢=
6¢3
)&
inputsÿÿÿÿÿÿÿÿÿ`À
p 

 
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿ`À
 
0__inference_functional_1_layer_call_fn_107946612×l9:@ABCPQWXYZghnopq~¨©²³¹º»¼ÅÆÏÐÖ×ØÙâãìíóôõöÿD¢A
:¢7
-*

imageInputÿÿÿÿÿÿÿÿÿ`À
p

 
ª "!ÿÿÿÿÿÿÿÿÿ`À
0__inference_functional_1_layer_call_fn_107946919×l9:@ABCPQWXYZghnopq~¨©²³¹º»¼ÅÆÏÐÖ×ØÙâãìíóôõöÿD¢A
:¢7
-*

imageInputÿÿÿÿÿÿÿÿÿ`À
p 

 
ª "!ÿÿÿÿÿÿÿÿÿ`À
0__inference_functional_1_layer_call_fn_107947787Ól9:@ABCPQWXYZghnopq~¨©²³¹º»¼ÅÆÏÐÖ×ØÙâãìíóôõöÿ@¢=
6¢3
)&
inputsÿÿÿÿÿÿÿÿÿ`À
p

 
ª "!ÿÿÿÿÿÿÿÿÿ`À
0__inference_functional_1_layer_call_fn_107947920Ól9:@ABCPQWXYZghnopq~¨©²³¹º»¼ÅÆÏÐÖ×ØÙâãìíóôõöÿ@¢=
6¢3
)&
inputsÿÿÿÿÿÿÿÿÿ`À
p 

 
ª "!ÿÿÿÿÿÿÿÿÿ`Àñ
N__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_107944025R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "H¢E
>;
04ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 É
3__inference_max_pooling2d_1_layer_call_fn_107944031R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ";84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿñ
N__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_107944141R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "H¢E
>;
04ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 É
3__inference_max_pooling2d_2_layer_call_fn_107944147R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ";84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿñ
N__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_107944257R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "H¢E
>;
04ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 É
3__inference_max_pooling2d_3_layer_call_fn_107944263R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ";84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿï
L__inference_max_pooling2d_layer_call_and_return_conditional_losses_107943909R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "H¢E
>;
04ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Ç
1__inference_max_pooling2d_layer_call_fn_107943915R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ";84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ²
F__inference_re_lu_1_layer_call_and_return_conditional_losses_107948251h7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ0` 
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ0` 
 
+__inference_re_lu_1_layer_call_fn_107948256[7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ0` 
ª " ÿÿÿÿÿÿÿÿÿ0` ²
F__inference_re_lu_2_layer_call_and_return_conditional_losses_107948408h7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ0@
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ0@
 
+__inference_re_lu_2_layer_call_fn_107948413[7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ0@
ª " ÿÿÿÿÿÿÿÿÿ0@´
F__inference_re_lu_3_layer_call_and_return_conditional_losses_107948565j8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿ
 
+__inference_re_lu_3_layer_call_fn_107948570]8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ
ª "!ÿÿÿÿÿÿÿÿÿ´
F__inference_re_lu_4_layer_call_and_return_conditional_losses_107948722j8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿ
 
+__inference_re_lu_4_layer_call_fn_107948727]8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ
ª "!ÿÿÿÿÿÿÿÿÿ´
F__inference_re_lu_5_layer_call_and_return_conditional_losses_107948892j8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿ
 
+__inference_re_lu_5_layer_call_fn_107948897]8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ
ª "!ÿÿÿÿÿÿÿÿÿ²
F__inference_re_lu_6_layer_call_and_return_conditional_losses_107949062h7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ0@
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ0@
 
+__inference_re_lu_6_layer_call_fn_107949067[7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ0@
ª " ÿÿÿÿÿÿÿÿÿ0@²
F__inference_re_lu_7_layer_call_and_return_conditional_losses_107949232h7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ0` 
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ0` 
 
+__inference_re_lu_7_layer_call_fn_107949237[7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ0` 
ª " ÿÿÿÿÿÿÿÿÿ0` ´
F__inference_re_lu_8_layer_call_and_return_conditional_losses_107949402j8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ`À
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿ`À
 
+__inference_re_lu_8_layer_call_fn_107949407]8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ`À
ª "!ÿÿÿÿÿÿÿÿÿ`À²
D__inference_re_lu_layer_call_and_return_conditional_losses_107948094j8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ`À
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿ`À
 
)__inference_re_lu_layer_call_fn_107948099]8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ`À
ª "!ÿÿÿÿÿÿÿÿÿ`À
'__inference_signature_wrapper_107947054òl9:@ABCPQWXYZghnopq~¨©²³¹º»¼ÅÆÏÐÖ×ØÙâãìíóôõöÿJ¢G
¢ 
@ª=
;

imageInput-*

imageInputÿÿÿÿÿÿÿÿÿ`À"6ª3
1
final(%
finalÿÿÿÿÿÿÿÿÿ`ÀÀ
R__inference_tf_op_layer_RealDiv_layer_call_and_return_conditional_losses_107947926j8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ`À
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿ`À
 
7__inference_tf_op_layer_RealDiv_layer_call_fn_107947931]8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ`À
ª "!ÿÿÿÿÿÿÿÿÿ`À¼
N__inference_tf_op_layer_Sub_layer_call_and_return_conditional_losses_107947937j8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ`À
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿ`À
 
3__inference_tf_op_layer_Sub_layer_call_fn_107947942]8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ`À
ª "!ÿÿÿÿÿÿÿÿÿ`À
S__inference_tf_op_layer_concat_6_layer_call_and_return_conditional_losses_107948734°~¢{
t¢q
ol
=:
inputs/0,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
+(
inputs/1ÿÿÿÿÿÿÿÿÿ
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿ
 à
8__inference_tf_op_layer_concat_6_layer_call_fn_107948740£~¢{
t¢q
ol
=:
inputs/0,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
+(
inputs/1ÿÿÿÿÿÿÿÿÿ
ª "!ÿÿÿÿÿÿÿÿÿ
S__inference_tf_op_layer_concat_7_layer_call_and_return_conditional_losses_107948904®|¢y
r¢o
mj
<9
inputs/0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
*'
inputs/1ÿÿÿÿÿÿÿÿÿ0@
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿ0
 Þ
8__inference_tf_op_layer_concat_7_layer_call_fn_107948910¡|¢y
r¢o
mj
<9
inputs/0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
*'
inputs/1ÿÿÿÿÿÿÿÿÿ0@
ª "!ÿÿÿÿÿÿÿÿÿ0
S__inference_tf_op_layer_concat_8_layer_call_and_return_conditional_losses_107949074­|¢y
r¢o
mj
<9
inputs/0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
*'
inputs/1ÿÿÿÿÿÿÿÿÿ0` 
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ0`@
 Ý
8__inference_tf_op_layer_concat_8_layer_call_fn_107949080 |¢y
r¢o
mj
<9
inputs/0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
*'
inputs/1ÿÿÿÿÿÿÿÿÿ0` 
ª " ÿÿÿÿÿÿÿÿÿ0`@
S__inference_tf_op_layer_concat_9_layer_call_and_return_conditional_losses_107949244¯}¢z
s¢p
nk
<9
inputs/0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
+(
inputs/1ÿÿÿÿÿÿÿÿÿ`À
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿ`À 
 ß
8__inference_tf_op_layer_concat_9_layer_call_fn_107949250¢}¢z
s¢p
nk
<9
inputs/0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
+(
inputs/1ÿÿÿÿÿÿÿÿÿ`À
ª "!ÿÿÿÿÿÿÿÿÿ`À â
I__inference_upsample_6_layer_call_and_return_conditional_losses_107944405¨©J¢G
@¢=
;8
inputs,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "@¢=
63
0,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 º
.__inference_upsample_6_layer_call_fn_107944415¨©J¢G
@¢=
;8
inputs,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "30,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿá
I__inference_upsample_7_layer_call_and_return_conditional_losses_107944557ÅÆJ¢G
@¢=
;8
inputs,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 ¹
.__inference_upsample_7_layer_call_fn_107944567ÅÆJ¢G
@¢=
;8
inputs,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@à
I__inference_upsample_8_layer_call_and_return_conditional_losses_107944709âãI¢F
?¢<
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 ¸
.__inference_upsample_8_layer_call_fn_107944719âãI¢F
?¢<
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ à
I__inference_upsample_9_layer_call_and_return_conditional_losses_107944861ÿI¢F
?¢<
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 ¸
.__inference_upsample_9_layer_call_fn_107944871ÿI¢F
?¢<
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ