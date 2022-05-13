
import pandas as pd
import random as rd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.cm as cm
import pickle
import random
import math
class neuron:
        def __init__(self,weights):
            self.weights = weights
            self.outputneuron=None
            self.s=None
            self.errorneuron=None
            self.deltaweights=[random.random()/2 for i in range(len(self.weights))].copy()
            self.deltaoldweights=[0 for i in range(len(self.weights))].copy()

        def S(self,inputs):
            return self.weights[0]+np.dot(self.weights[1:],inputs)

        def activation_function(self,inputs):
            outputneuron=self.f(self.S(inputs))
            return outputneuron
        #bacKpropagation
        def f(self,x):
            return 1/(1+math.exp(-x))
        def derivation_function(self,s):
            return self.f(s)*(1-self.f(s))

        def update_weights_ofsingle_neuronj(self):
            for k in range(len(self.deltaweights)):
                wjk=self.weights[k]
                self.weights[k]=wjk+self.deltaweights[k]


class layer:
    def __init__(self,nbrof_neurons,nbrof_inputs):
        self.numberof_neurons = nbrof_neurons
        self.numerof_inputs= nbrof_inputs
        self.my_neurons=layer.generate_my_neurons(nbrof_neurons,nbrof_inputs)
        self.inputstolayer=None
        self.outputsoflayer=None

    def generate_my_neurons(nbrof_neurons,nbrof_inputs):
        list_of_neurons=[]
        #a=math.sqrt(6)/math.sqrt(nbrof_inputs+nbrof_neurons)
        a=math.sqrt(3)/math.sqrt(nbrof_inputs)
        #a=1
        #a=math.sqrt(6)/math.sqrt(nbrof_inputs+nbrof_neurons)
        for i in range(nbrof_neurons):
            weights_of_neuron=[np.random.uniform(-a,a) for i in range(nbrof_inputs+1)].copy()
            #weights_of_neuron=[1 for i in range(nbrof_inputs+1)].copy()
            new_neuron=neuron(weights_of_neuron)
            list_of_neurons.append(new_neuron)
        return list_of_neurons

    def generate_my_outputs(self,nbrof_neurons):
        outputslayer=[]
        for i in range(nbrof_neurons):
            output=self.my_neurons[i].activation_function(self.inputstolayer)
            outputslayer.append(output)
            self.my_neurons[i].outputneuron=output
            self.my_neurons[i].s=self.my_neurons[i].S(self.inputstolayer)

        return outputslayer
    
class network:
    def __init__(self,nbrof_layers,nbrof_neuronsinlayer,Inputs,number_of_classes):
        self.number_of_layers = nbrof_layers
        self.number_of_classes=number_of_classes
        self.nbrof_neuronsinlayer=nbrof_neuronsinlayer
        self.input_layer=Inputs.copy()
        self.myhidden_layers=self.generatemylayers(nbrof_layers,nbrof_neuronsinlayer)
        self.output_layer=self.generete_output_layer(number_of_classes)
        self.desiredoutput=None
        #self.Activate_network(nbrof_neuronsinlayer)
    def generatemylayers(self,nbrof_layers,nbrof_neuronsinlayer):
        list_of_layers=[]
        for i in range (nbrof_layers):
            if len(list_of_layers)==0:
                nbrof_inputs=len(self.input_layer)
            else:
                nbrof_inputs=list_of_layers[i-1].numberof_neurons
            new_layer=layer(nbrof_neuronsinlayer[i],nbrof_inputs)
            list_of_layers.append(new_layer)
        return list_of_layers

    def generete_output_layer(self,number_of_classes):
        if len(self.myhidden_layers)==0:
            print("no hiddenlayer")
            nbrof_inputstooutlayer=len(self.input_layer)
        else:
            before_output_layer=self.myhidden_layers[len(self.myhidden_layers)-1]
            nbrof_inputstooutlayer=before_output_layer.numberof_neurons
        output_layer=layer(number_of_classes,nbrof_inputstooutlayer)
        return output_layer

    def Activate_network(self,nbrof_neuronsinlayer):
        if self.number_of_layers==0:
            print("no hidden layers")
            self.output_layer.inputstolayer=self.input_layer.copy()
            self.output_layer.outputsoflayer=self.output_layer.generate_my_outputs(self.number_of_classes).copy()
        else:
            self.myhidden_layers[0].inputstolayer=self.input_layer.copy()
            self.myhidden_layers[0].outputsoflayer=self.myhidden_layers[0].generate_my_outputs(nbrof_neuronsinlayer[0]).copy()
            for i in range(1,self.number_of_layers):
                self.myhidden_layers[i].inputstolayer= self.myhidden_layers[i-1].outputsoflayer.copy()
                self.myhidden_layers[i].outputsoflayer=self.myhidden_layers[i].generate_my_outputs(nbrof_neuronsinlayer[i]).copy()

            self.output_layer.inputstolayer=self.myhidden_layers[self.number_of_layers-1].outputsoflayer.copy()
            self.output_layer.outputsoflayer=self.output_layer.generate_my_outputs(self.number_of_classes).copy()


    #Backpropagation:
    def update_error_output_layer(self,desired_outputs):
        H=0.9
        L=0.1
        n=self.number_of_layers
        learning_rate=0.0008
        skiped=[]
        for i in range(self.number_of_classes):
            if desired_outputs[i]==1 and self.output_layer.my_neurons[i].outputneuron>H:
                skiped.append(i)
                self.output_layer.my_neurons[i].errorneuron=0
                continue
            elif desired_outputs[i]==0 and self.output_layer.my_neurons[i].outputneuron<L:
                skiped.append(i)
                self.output_layer.my_neurons[i].errorneuron=0
                continue
            else:
                ei=desired_outputs[i]-self.output_layer.my_neurons[i].outputneuron
                si=self.output_layer.my_neurons[i].s
                self.output_layer.my_neurons[i].errorneuron=self.output_layer.my_neurons[i].derivation_function(si)*ei
                self.calculatedeltaj(self.output_layer.my_neurons[i],self.myhidden_layers[n-1],learning_rate)
        return skiped
            

    def update_error_hidden_layers(self):
        n=self.number_of_layers
        if n==0:
            print("no hidden layers")
            return
        self.update_error_singlehidden_layer(self.myhidden_layers[n-1],self.output_layer,self.nbrof_neuronsinlayer[n-1],n-1)
        for l in range(n-2,-1,-1):
            self.update_error_singlehidden_layer(self.myhidden_layers[l],self.myhidden_layers[l+1],self.nbrof_neuronsinlayer[l],l)
        return

    def update_error_singlehidden_layer(self,current_layer,following_layer,nbrofneuronsinlayer,l):
        learning_rate=0.0008
        for j in range(nbrofneuronsinlayer):
            dotprod=self.dot_weights_errors(j,following_layer)#decalge du bias
            sj=current_layer.my_neurons[j].s
            current_layer.my_neurons[j].errorneuron=current_layer.my_neurons[j].derivation_function(sj)*dotprod
            if l==0:
                self.calculatedeltaj_first_layer(current_layer.my_neurons[j],learning_rate)
            else:
                self.calculatedeltaj(current_layer.my_neurons[j],self.myhidden_layers[j-1],learning_rate)
        return

    def dot_weights_errors(self,j,following_layer):
        j=j+1#decalge du bias
        dotproduct=0
        for i in range(following_layer.numberof_neurons):
            wijerrori=following_layer.my_neurons[i].errorneuron*following_layer.my_neurons[i].weights[j]
            dotproduct=dotproduct+wijerrori
        return dotproduct
    #uppdate weights of neuron j:
    def calculatedeltaj(self,neuronj,previous_layer,learning_rate):
        alpha=0.8
        ej=neuronj.errorneuron#add TO SAVE OLD WEIGHTS
        neuronj.deltaweights[0]=learning_rate*ej+alpha*neuronj.deltaoldweights[0]
        neuronj.deltaoldweights[0]=neuronj.deltaweights[0]
        for k in range(0,previous_layer.numberof_neurons):
            hk=previous_layer.outputsoflayer[k]
            neuronj.deltaweights[k+1]=learning_rate*ej*hk+alpha*neuronj.deltaoldweights[k+1]
            neuronj.deltaoldweights[k+1]=neuronj.deltaweights[k+1]
        return
    def calculatedeltaj_first_layer(self,neuronj,learning_rate):
        alpha=0.8
        ej=neuronj.errorneuron#add TO SAVE OLD WEIGHTS
        neuronj.deltaweights[0]=learning_rate*ej+alpha*neuronj.deltaoldweights[0]
        neuronj.deltaoldweights[0]=neuronj.deltaweights[0]
        for k in range(0,len(self.input_layer)):
            hk=self.input_layer[k]
            neuronj.deltaweights[k+1]=learning_rate*ej*hk+alpha*neuronj.deltaoldweights[k+1]
            neuronj.deltaoldweights[k+1]=neuronj.deltaweights[k+1]
        return

    def backpropagate(self):
        skiped=self.update_error_output_layer(self.desiredoutput) 
        self.update_error_hidden_layers()          
        for i in range(self.number_of_classes):
            if (i in skiped):
                continue
            else:
                self.output_layer.my_neurons[i].update_weights_ofsingle_neuronj() 
        for i in range(self.number_of_classes):
            self.output_layer.my_neurons[i].update_weights_ofsingle_neuronj() 
        for layer in self.myhidden_layers:
            for neuron in layer.my_neurons:
                neuron.update_weights_ofsingle_neuronj()
        #print("finish backpropagation")
        return
    def classify(self):
        index=np.argmax(self.output_layer.outputsoflayer)
        return index
    



# In[2]:


##
#Import shuffled images
Inputsst=pd.read_csv("Imagestrainshuffled.txt",sep=' ',header=None)
teaching_input=pd.read_csv("Labelstrainshuffled.txt",sep=' ',header=None)
Inputsst=Inputsst.values
teaching_input=teaching_input.values.tolist()
lg_data=len(teaching_input)
##Import testing images
Inputstestu=pd.read_csv("Imagestestshuffled.txt",sep=' ',header=None)
testing_input=pd.read_csv("Labelstestshuffled.txt",sep=' ',header=None)
Inputstestu=Inputstestu.values.tolist()
testing_input=testing_input.values.tolist()
lg_test=len(testing_input)
#Imagestrainshuffled
#Labelstrainshuffled


# In[3]:


mean=np.mean(Inputsst)
std=np.std(Inputsst)
Inputs=(Inputsst-mean)/std


# In[4]:


#Inputs=Inputsst
mean=np.mean(Inputstestu)
std=np.std(Inputstestu)
Inputstest=(Inputstestu-mean)/std

# 
# # In[7]:
# 
# 
# net=network(1,[64],Inputs[0],10)
# desiredout=[0 for i in range(net.number_of_classes)]
# errors=[]
# errorstest=[]
# epochs=[]
# indexestest=[i for i in range(1000)]
# n=250
# indexes=[rd.randrange(0,lg_data) for i in range(n)]
# error=0
# errortest=0
# for j in indexes:
#         net.input_layer=Inputs[j].copy()
#         testout=teaching_input[j][0]
#         net.Activate_network(net.nbrof_neuronsinlayer)
#         output=net.classify()
#         if output-testout!=0:
#             error=error+1
# errors.append(error/n)
# epochs.append(0)
# for e in indexestest:
#     net.input_layer=Inputstest[e].copy()
#     testouttest=testing_input[e][0]
#     net.Activate_network(net.nbrof_neuronsinlayer)
#     outputtest=net.classify()
#     if outputtest-testouttest!=0:
#         errortest=errortest+1
#     #print(abs(output-testout))
# errorstest.append(errortest/1000)
# for epoch in range(300):
#     error=0
#     errortest=0
#     n=450
#     indexes=[rd.randrange(0,lg_data) for i in range(n)]
#     for i in indexes:
#         desiredout=[0 for o in range(net.number_of_classes)]
#         #print(i)
#         #print(teaching_input[i][0])
#         desiredout[teaching_input[i][0]]=1
#         net.input_layer=Inputs[i].copy()
#         net.desiredoutput=desiredout.copy()
#         net.Activate_network(net.nbrof_neuronsinlayer)
#         net.backpropagate()
#     if epoch%10==0 and epoch!=0:
#         for j in indexes:
#             net.input_layer=Inputs[j].copy()
#             testout=teaching_input[j][0]
#             net.Activate_network(net.nbrof_neuronsinlayer)
#             output=net.classify()
#             if output-testout!=0:
#                 error=error+1
#         errors.append(error/n)
#         epochs.append(epoch)
#         
#         for e in indexestest:
#             net.input_layer=Inputstest[e].copy()
#             testouttest=testing_input[e][0]
#             net.Activate_network(net.nbrof_neuronsinlayer)
#             outputtest=net.classify()
#             if outputtest-testouttest!=0:
#                 errortest=errortest+1
#             #print(abs(output-testout))
#         errorstest.append(errortest/1000)
#     if epoch%10==0:
#         plt.figure()
#         plt.plot(epochs,errors)
#         plt.plot(epochs,errorstest)
#         plt.show()
# 
#     print(epoch)
#         
# plt.figure()
# plt.plot(epochs,errors)
# plt.plot(epochs,errorstest)
# plt.show()
# 
# 
# 
# # In[8]:
# 
# 
# def save_object(obj, filename):
#     with open(filename, 'wb') as outp:  # Overwrites any existing file.
#         pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)
# 
# 
# # In[9]:
# 
# 
# save_object(errors,'errors2.pkl')
# save_object(net, 'network2.pkl')
# save_object(epochs,'epochs2.pkl')
# save_object(errorstest,'errorstest2.pkl')
# 
# 
# # In[10]:
# 
# 
# 
# 
# # In[11]:

####Testing the performance and loading the model
import pandas as pd
import random as rd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.cm as cm
import pickle
import random
import math

with open('network2.pkl', 'rb') as inp:
    nettrained = pickle.load(inp)

with open('errors3.pkl', 'rb') as inp:
    errors = pickle.load(inp)

with open('errorstest3.pkl', 'rb') as inp:
    errorstest = pickle.load(inp)
with open('epochs3.pkl', 'rb') as inp:
    epochs = pickle.load(inp)
#     
# In[12]:
plt.figure()
plt.plot(epochs,errors,label='Training error fraction')
plt.plot(epochs,errorstest,label='Testing errors fraction')
plt.title("Error fraction for Training and Testing data each 10th epochs")
plt.legend()
plt.xlabel('epochs')
plt.ylabel('Error fraction')
plt.show()


Ypred=[]
Ypredtest=[]
for j in range (lg_data):
    nettrained.input_layer=Inputs[j].copy()
    testout=teaching_input[j][0]
    nettrained.Activate_network(nettrained.nbrof_neuronsinlayer)
    output=nettrained.classify()
    Ypred.append(output)

for e in range (lg_test):
    nettrained.input_layer=Inputstest[e].copy()
    testouttest=testing_input[e][0]
    nettrained.Activate_network(nettrained.nbrof_neuronsinlayer)
    outputtest=nettrained.classify()
    Ypredtest.append(outputtest)
            
    


# In[14]:


print(len(Ypredtest))
print(len(Ypred))


# In[15]:


from sklearn import metrics
cm=metrics.confusion_matrix(teaching_input,Ypred)
print(cm)


# In[26]:


import seaborn as sns
df_cm = pd.DataFrame(cm, index = [i for i in "0123456789"],
                  columns = [i for i in "0123456789"])
plt.figure(figsize = (10,6))
heatmap=sns.heatmap(df_cm, annot=True,cmap='BrBG', fmt='g')
bottom, top = heatmap.get_ylim()
heatmap.set_ylim(bottom + 0.5, top - 0.5)
plt.xlabel("Predicted labels")
plt.ylabel("True labels") 
plt.title("The confusion matrix of the training data")
plt.show()

# In[22]:


from sklearn import metrics
cmtest=metrics.confusion_matrix(testing_input,Ypredtest)
print(cmtest)


# In[25]:


import seaborn as sns
df_cmtest= pd.DataFrame(cmtest, index = [i for i in "0123456789"],
                  columns = [i for i in "0123456789"])
plt.figure(figsize = (10,7))
heatmap=sns.heatmap(df_cmtest, annot=True,cmap='BrBG', fmt='g')
bottom, top = heatmap.get_ylim()
heatmap.set_ylim(bottom + 0.5, top - 0.5)
plt.xlabel("Predicted labels")
plt.ylabel("True labels") 
plt.title("The confusion matrix of the testing data")
plt.show()





