import math
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa


class Block(tf.keras.layers.Layer):
  def __init__(self, name,sequences=[],operation='scale', scale=0,delta=0,translateX=0,translateY=0,filter_shape=3,rotation=0,sharpen_factor=1,sinX=1,sinY=1):
    super(Block, self).__init__(name=name)
    
    self.sequences=sequences
    self.operation=operation
    self.scale = scale
    self.delta = delta
    self.translateX = translateX
    self.translateY = translateY
    self.filter_shape = filter_shape
    self.rotation = rotation
    self.sharpen_factor = sharpen_factor
    self.sinX = sinX
    self.sinY = sinY

  def call(self, inputs):
    
    if self.operation=='block':  
        print('operation: block')
        
        #diag = tf.cast(tf.linalg.diag(self.sequences), tf.float32)
        #inputs += tf.ones(tf.shape(inputs))
        #scaled = tf.tensordot(diag,tf.transpose(inputs,[1,2,3,0]),axes=1)
        #inputs -= tf.ones(tf.shape(inputs))
        out = tf.zeros(tf.shape(inputs))
        
    elif self.operation=='scale':
        x = inputs.shape[-1]
        x_new = int(inputs.shape[-1]+self.scale)
        print(f'operation: scale: {x} -> {x_new}')
        base_t = tf.transpose(inputs,[1,2,3,0])
        base_re = tf.image.resize(base_t, [x_new,x_new], method='gaussian')
        base_crop = tf.image.resize_with_crop_or_pad(base_re,x,x)
        out = tf.transpose(base_crop,[3,0,1,2])
        
    elif self.operation=='invert':
        print('operation: invert')
        out = -inputs
        
    elif self.operation=='shuffle':
        print('operation: shuffle')
        outs_t = tf.transpose(inputs,[1,2,3,0])
        r_t = tf.random.shuffle(outs_t,seed=1)
        out = tf.transpose(r_t,[3,0,1,2])
        
    elif self.operation =='brightness':
        print(f'operation: brightness: {self.delta}')
        out = tf.image.adjust_brightness(inputs, delta=self.delta)
        
    elif self.operation =='translate':
        print(f'operation: translate[{self.translateX}, {self.translateY}]')
        trans_1 = tf.transpose(inputs, [1,2,3,0])
        rot = tfa.image.translate(trans_1,[self.translateX,self.translateY], interpolation='bilinear', fill_mode='reflect')
        out = tf.transpose(rot,[3,0,1,2])  
        out.set_shape(inputs.shape)
        
    elif self.operation=='vanish':
        print('operation: vanish')
        selected = -self.sequences+1
        pattern = np.where(selected==1)
        based_t = tf.transpose(inputs,[1,2,3,0])
        based_sliced = tf.gather(based_t,pattern[0])
        lens = tf.shape(based_sliced)[0]
        based_repeat = tf.repeat(based_sliced,int(512/lens),axis=0)
        
        based_full = tf.concat([based_repeat,based_sliced[:512%lens]], axis=0)
        out = tf.transpose(based_full,[3,0,1,2])
        
    elif self.operation=='mean_filter':
        print('operation: mean_filter')
        trans_1 = tf.transpose(inputs, [1,2,3,0])
        filtered = tfa.image.mean_filter2d(trans_1,filter_shape = (self.filter_shape,self.filter_shape))
        out = tf.transpose(filtered,[3,0,1,2])  
        out.set_shape(inputs.shape)
        
    elif self.operation=='rotate':
        print(f'operation: rotate {self.rotation}')
        trans_1 = tf.transpose(inputs, [1,2,3,0])
        rotated = tfa.image.rotate(trans_1, tf.constant(self.rotation),fill_mode='reflect')
        out = tf.transpose(rotated,[3,0,1,2])  
        out.set_shape(inputs.shape)
        
    elif self.operation=='sharpen':
        print(f'operation: sharpen {self.sharpen_factor}')
        trans_1 = tf.transpose(inputs, [1,2,3,0])
        print(trans_1.shape)
        sharpen = tfa.image.sharpness(trans_1, tf.constant(self.sharpen_factor,dtype=tf.float32))
        out = tf.transpose(sharpen,[3,0,1,2])  
        out.set_shape(inputs.shape)
        
    elif self.operation=='erosion':
        print('operation: erosion')
        based_t = tf.transpose(inputs,[1,2,3,0])
        based_ero = tf.nn.erosion2d(based_t,
                                    filters=tf.zeros((3,3,tf.shape(inputs)[0])),
                                    strides=(1,1,1,1),
                                    padding='SAME',
                                    data_format='NHWC',
                                    dilations=(1,1,1,1))
        out = tf.transpose(based_ero,[3,0,1,2])
        
    elif self.operation=='dilation':
        print('operation: dilation')
        based_t = tf.transpose(inputs,[1,2,3,0])
        based_ero = tf.nn.dilation2d(based_t,
                                    filters=tf.zeros((3,3,tf.shape(inputs)[0])),
                                    strides=(1,1,1,1),
                                    padding='SAME',
                                    data_format='NHWC',
                                    dilations=(1,1,1,1))
        out = tf.transpose(based_ero,[3,0,1,2])
        
    elif self.operation=='mirrorY':
        print('operation: mirrorY')
        based_t = tf.transpose(inputs,[1,2,3,0])
        
        mirror_full = tfa.image.rotate(based_t, tf.constant(np.pi),fill_mode='reflect')
        x = int(based_t.shape[1]/2)
        print(x)
        mirror = tf.concat([based_t[:,:x,:,:],mirror_full[:,x:,:,:]],axis=1)
        out = tf.transpose(mirror,[3,0,1,2])
        
    elif self.operation=='mirrorX':
        print('operation: mirrorX')
        based_t = tf.transpose(inputs,[1,2,3,0])
        
        mirror_full = tfa.image.rotate(based_t, tf.constant(np.pi),fill_mode='reflect')
        x = int(based_t.shape[1]/2)
        print(x)
        mirror = tf.concat([based_t[:x,:,:,:],mirror_full[x:,:,:,:]],axis=0)
        out = tf.transpose(mirror,[3,0,1,2])
        
    elif self.operation=='sin_disrupt':
        w = inputs.shape[2]
        h = inputs.shape[3]
        base = np.zeros((1,512,h,w))
        for x in range(w):
            for y in range(h):
                base[:,:,x,y] += (np.sin((x-4-self.translateY)/w*self.sinX))*4
                base[:,:,x,y] += (np.sin((y-2-self.translateX)/w*self.sinY))*4
        out = tf.convert_to_tensor(base,dtype=tf.float32)
        
    else:
        print('no operation!!!!')
        
    print(f'operation applied to {np.count_nonzero(self.sequences)} layers')
    ones = tf.ones(tf.shape(inputs))
    diag = tf.cast(tf.linalg.diag(self.sequences), tf.float32)
    scaled = tf.tensordot(diag,tf.transpose(ones,[1,2,3,0]),axes=1)
    mask = tf.cast(tf.transpose(scaled,[3,0,1,2]),dtype=tf.bool)
    
    filtered_out = tf.where(mask,out,inputs)
    return filtered_out

  def build(self,input_shape):
      self.compute_output_shape(input_shape)
      
  def compute_output_shape(self, inputs):
    self._spatial_output_shape(inputs)
    return inputs

  def _spatial_output_shape(self, spatial_input_shape):
    return spatial_input_shape



class Scale(tf.keras.layers.Layer):
  def __init__(self, name, scale):
    super(Scale, self).__init__(name=name)
    self.scale = scale
    
  def call(self, inputs):
    base_t = tf.transpose(inputs,[1,2,3,0])
    x = inputs.shape[-1]
    x_new = int(inputs.shape[-1]+self.scale)
    base_re = tf.image.resize(base_t, [x_new,x_new], method='gaussian')
    print(x)
    print(x_new)
    base_crop = tf.image.resize_with_crop_or_pad(base_re,x,x)
    base_crop_t = tf.transpose(base_crop,[3,0,1,2])
    return base_crop_t



class RepeatX(tf.keras.layers.Layer):
  def __init__(self, name):
    super(RepeatX, self).__init__(name=name)

  def call(self, inputs):
    print(inputs.shape)
    #out = tf.repeat(inputs[:,:,:int(inputs.shape[2]/2),:],repeats=2,axis=2)
    out = tf.concat([inputs[:,:,:,:2],
                     inputs[:,:,:,:2]
                     ],axis=3)
    print(out.shape)
    return out

  def build(self,input_shape):
      self.compute_output_shape(input_shape)
      
  def compute_output_shape(self, inputs):
    self._spatial_output_shape(inputs)
    return inputs

  def _spatial_output_shape(self, spatial_input_shape):
    return spatial_input_shape

class RepeatY(tf.keras.layers.Layer):
  def __init__(self, name):
    super(RepeatY, self).__init__(name=name)

  def call(self, inputs):
    print(inputs.shape)
    #out = tf.repeat(inputs[:,:,:int(inputs.shape[2]/2),:],repeats=2,axis=2)
    out = tf.concat([inputs[:,:,:1,:],
                     inputs[:,:,:1,:],
                     inputs[:,:,:1,:],
                     inputs[:,:,:1,:]
                     ],axis=2)
    print(out.shape)
    return out

  def build(self,input_shape):
      self.compute_output_shape(input_shape)
      
  def compute_output_shape(self, inputs):
    self._spatial_output_shape(inputs)
    return inputs

  def _spatial_output_shape(self, spatial_input_shape):
    return spatial_input_shape

class Shuffle(tf.keras.layers.Layer):
  def __init__(self, name):
    super(Shuffle, self).__init__(name=name)

  def call(self, inputs):
    outs_t = tf.transpose(inputs,[1,2,3,0])
    r_t = tf.random.shuffle(outs_t,seed=1)
    r = tf.transpose(r_t,[3,0,1,2])
    return r

class Brightness(tf.keras.layers.Layer):
  def __init__(self, name,delta=0):
    super(Brightness, self).__init__(name=name)
    self.delta = delta

  def call(self, inputs):
    print('b')
    return tf.image.adjust_brightness(inputs, delta=self.delta)


class Sin_disrupt(tf.keras.layers.Layer):
  def __init__(self, name):
    super(Sin_disrupt, self).__init__(name=name)

  def call(self, inputs):
    
    w = inputs.shape[2]
    h = inputs.shape[3]
    base = np.zeros((1,512,h,w))
    for x in range(w):
        for y in range(h):
            base[:,:,x,y] += (np.sin(x/w))/20
            base[:,:,x,y] += (np.sin(y/w))/20
    return tf.convert_to_tensor(base,dtype=tf.float32)

  def build(self,input_shape):
      self.compute_output_shape(input_shape)
      
  def compute_output_shape(self, inputs):
    self._spatial_output_shape(inputs)
    return inputs

  def _spatial_output_shape(self, spatial_input_shape):
    return spatial_input_shape


class Rotate(tf.keras.layers.Layer):
  def __init__(self, name, rotation=0):
    super(Rotate, self).__init__(name=name)
    self.rotation=rotation

  def call(self, inputs):
    '''
    trans_1 = tf.transpose(inputs, [0,2,3,1])
    rot = tf.image.rot90(trans_1,k=1)
    trans_2 = tf.transpose(rot,[0,3,1,2])
    '''
    trans_1 = tf.transpose(inputs, [1,2,3,0])
    rotated = tfa.image.rotate(trans_1, tf.constant(self.rotation),fill_mode='constant')
    trans_2 = tf.transpose(rotated,[3,0,1,2])  
    trans_2.set_shape(inputs.shape)
    
    return trans_2

class Invert(tf.keras.layers.Layer):
  def __init__(self, name):
    super(Invert, self).__init__(name=name)

  def call(self, inputs):
    return -inputs


class Translate(tf.keras.layers.Layer):
  def __init__(self, name, translateX=0,translateY=0):
    super(Translate, self).__init__(name=name)
    self.translateX = translateX
    self.translateY = translateY
    
  def call(self, inputs):
    trans_1 = tf.transpose(inputs, [1,2,3,0])
    rot = tfa.image.translate(trans_1,[self.translateX,self.translateY], interpolation='nearest', fill_mode='reflect')
    trans_2 = tf.transpose(rot,[3,0,1,2])  
    trans_2.set_shape(inputs.shape)
    return trans_2

  def build(self,input_shape):
      self.compute_output_shape(input_shape)
      
  def compute_output_shape(self, inputs):
    self._spatial_output_shape(inputs)
    return inputs

  def _spatial_output_shape(self, spatial_input_shape):
    return spatial_input_shape