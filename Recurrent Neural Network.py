import numpy as np 
""" 
basic implementation of Recurrent Neural Networks from scrach
to train model to learn to add any number pair when given in binary arrayed format
devloper-->sayaneree paria
"""
class RecurrentNeuralNetwork:
    
    def __init__(self,hidden_size=10): 
        """hidden_size is number of neurons in hidden layer"""  
        self.hidden_size=hidden_size
        self.activation={"sigmoid":(self.sigmoid,self.sig_grad),
                            "RELU":(self.RELU,self.RELU_grad),
                            "tanh":(self.tanh,self.tanh_grad)}
  
    def fit(self,X,Y):
        """input your training dataset
            X: input array 3D 
            Y: output arrray 3D 
        axis0- number of data data
        axis1 -oredered steps(time steps) of data
        axis2- input array for each step"""

        #add a slot for threshold weight in each inputs     
        X=np.append(X,np.ones((X.shape[0],X.shape[1],1)),axis=2)
        # store sizes of datasets
        self.input_size=X.shape[2]
        self.output_size=Y.shape[2]
        self.X=X
        self.Y=Y

    def tanh(self,x):
        """for hyperbolic tangent activation"""
        return np.tanh(x)

    def tanh_grad(self,x):
        """gradiant through tanh function"""
        return np.minimum(1-self.tanh(x)**2,1e2)
   
    def RELU(self,x):
        """for RELU activation"""
        return np.maximum(x,0)

    def RELU_grad(self,x):
        """gradient through RELU function"""
        return np.sign(x)

    def sigmoid(self,x):
        """sigmoid activation"""
        return 1/(1+np.exp(-x))

    def sig_grad(self,x):
        """gradiant through sigmoid function"""
        return x*(1-x)
    
    def train(self,rate=1,activation="sigmoid"):
        """train the model on the dataset provided , rate: learning rate"""
        
        activate,actv_grad=self.activation[activation]

        
        # initialise our weights randomly for hidden and output layers and recursion of previous layers
        hidden_weight=2*np.random.random((self.input_size,self.hidden_size))-1              
        output_weight=2*np.random.random((self.hidden_size,self.output_size))-1
        recurent_weight=2*np.random.random((self.hidden_size,self.hidden_size))-1       

        #terate through all data in dataset
        for i,X1 in enumerate(self.X):
            #corosponding output
            Y1=self.Y[i]

            #lists to store our outputs to help find gradients of all  timestep
            hidden_layers=list()
            output_gradients=list()

            #initially we set our feedback vector to zero
            hiddenlayer=np.zeros((1,self.hidden_size))
            hidden_layers.append(hiddenlayer)

            #keep track of error
            total_errors=0
            # forward propagate in time steps finding output of the RNN
            for time,X in enumerate(X1):
                # hidden state is function of both input of current time step and hidden state of previous time step
                #note we can also use other activation like RELU or tanh which may affect performanc
                
                hiddenlayer= activate(np.dot(X,hidden_weight)+np.dot(hidden_layers[-1],recurent_weight))
                outputlayer= activate(np.dot(hiddenlayer,output_weight))
                #calulate error
                error= Y1[time]-outputlayer
                total_errors+=np.abs(error[0,0])
                #gradient of output layer
                outputGradient=error*actv_grad(outputlayer)
                #we store the hidden layers and output gradients to calculate the gradients of weight vectors
                hidden_layers.append(np.atleast_2d(hiddenlayer))
                output_gradients.append(np.atleast_2d(outputGradient))

            #initialise all gradients zero
            output_weight_gradient=np.zeros_like(output_weight)
            hidden_weight_gradient=np.zeros_like(hidden_weight)
            recurent_weight_gradient=np.zeros_like(recurent_weight)

            #we use this to store the gradient of cost function (of future time) wrt  time steps (in current time) on which it depends  
            future_gradients=np.zeros(self.hidden_size)
            # iterate in reverse order, backpropagation through time!
            for time,X in enumerate(X1[::-1]):
                time=X1.shape[0]-time-1
                #recursively set current gradients and all future gradients linked to this time step
                hidden_layer_gradients=(np.dot(future_gradients,recurent_weight.T)+ np.dot(output_gradients[time],output_weight.T))*actv_grad(hidden_layers[time+1])
                #sum of gradients of error in each time step
                output_weight_gradient+=hidden_layers[time+1].T.dot(output_gradients[time])
                hidden_weight_gradient+=np.atleast_2d(X).T.dot(hidden_layer_gradients)
                recurent_weight_gradient+=np.dot(hidden_layers[time].T,hidden_layer_gradients)
                #use this in next iteration to set gradients linked to past
                future_gradients=hidden_layer_gradients
            # update out weights by the learning rate
            hidden_weight += rate * hidden_weight_gradient
            output_weight+=rate * output_weight_gradient
            recurent_weight += rate * recurent_weight_gradient
            # print error in intervals
            if i %1000==0:
                print("iteration: {0}\t\t error: {1}".format(i,total_errors))
        #we save our weights
        self.hidden_weight=hidden_weight
        self.output_weight=output_weight
        self.recurent_weight=recurent_weight

    def predict(self,X):
        """predict the output of X"""
        #add slot for thresholds
        X=np.append(X,np.ones((X.shape[0],X.shape[1],1)),axis=2)
        output=np.zeros((X.shape[0],X.shape[1],self.output_size))
        #set feedback to zero intially
        prev_hiddenlayer=np.zeros((1,self.hidden_size))
        #iterate through all input data and do pediction
        for j,X2 in enumerate(X):
            for time,X1 in enumerate(X2):
                        
                hiddenlayer= self.sigmoid(np.dot(X1,self.hidden_weight)+np.dot(prev_hiddenlayer,self.recurent_weight))
                outputlayer= self.sigmoid(np.dot(hiddenlayer,self.output_weight))
                output[j,time]=outputlayer
                prev_hiddenlayer=hiddenlayer
        
        return output

###we train RNN to learn how to add two numbers

# we generate 10,1000 random pair of numbers whose sum is below 2^8
max_val = 2**8
a=np.random.randint(0,high=max_val/2,size=(10000,2,1),dtype=np.uint8)
#convert to binary format
b= np.transpose(np.unpackbits(a, axis=2),(2,1,0))
#reverse order to keep LSB(least significant bit)first
b=b[::-1].transpose((2,0,1))
#sum the pairs with LSB first
sum=np.atleast_3d(np.unpackbits(np.sum(a,axis=1,dtype=np.uint8),axis=1).T[::-1].T)

#create instance of our model we will use 8 neurons in hidden layers it may be changed according to requirments
rnn=RecurrentNeuralNetwork(hidden_size=8)
#train on first 9980 data
rnn.fit(b[:9980],sum[:9980])
rnn.train(rate=1)
#print prediction for last 20 row wise
print(np.round(rnn.predict(b[9980:])).astype(int).transpose(2,0,1))
#and print the actual sums
print(sum[9980:].transpose(2,0,1))




                






            




