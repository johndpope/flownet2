#dumpyard
#depth estimation
flow_transposed=tf.transpose(flow[0,:,:,:], perm=[2, 0, 1])#[channels,height,width]
flow_transposed=tf.concat([flow_transposed, tf.zeros([1,height,width],dtype=tf.float32)],0)
x1=tf.tile(tf.constant([[i for i in range(0,width)]],dtype=tf.float32),[height,1])
y1=tf.tile(tf.constant([[i] for i in range(0,height)],dtype=tf.float32),[1,width])
z1=tf.ones([height,width],dtype=tf.float32)
x1=tf.expand_dims(x1, 0)
y1=tf.expand_dims(y1, 0)
z1=tf.expand_dims(z1, 0)
X=tf.concat([x1,y1,z1],0)
I=tf.constant([[1,0,0],[0,1,0],[0,0,1]],dtype=tf.float32)
R1=(k2*rotation*tf.matrix_inverse(k1)-I)
m=[]
for i in range(0, height):
    temp=tf.matmul(R1,tf.reshape(X[:,i,:],[3,width]))
    m.append(temp)
g=tf.stack(m,axis=1)
inv_t=1/(flow_transposed-g)
m=[]
for i in range(0, height):
    temp=tf.matmul(tf.matmul(translation,k2),tf.reshape(inv_t[:,i,:],[3,width]))
    m.append(temp)
depth=tf.stack(m,axis=1)

# depth=tf.matrix_inverse(flow_transposed-(k2*rotation*tf.matrix_inverse(k1)-I)*X)*(k2*translation)

# depth=lmbspecialops.flow_to_depth(flow=flow_transposed,intrinsics=intrinsics_1,rotation=rotation,translation=translation,rotation_format='matrix')
depth=tf.transpose(depth, perm=[1, 2,0])
depth=tf.expand_dims(depth, 0)
depth=tf.tile(depth,[1,1,1,3])
