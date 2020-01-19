# Solving-the-binding-problem-in-a-neural-network-model-of-serial-recall

A replication of the Immediate Serial Recall model of Botvinick & Plaut (2006)
and the implementation of the dual model, along with the simulations used in 
the thesis under the supervision of Ivan Vankov, PhD.

Brief desctiption of the project:

Short-term memory (STM) has been regarded as one of the key properties of the human
cognitive system due to its ubiquitous role in domains such as language comprehension,
problem-solving, or long-term learning. A key aspect of STM is the problem of serial order.
This thesis identifies the binding problem as  important problem in serial recall, which  
consists of explaining how the cognitive system associates a particular item and the position
on which it is to be remembered. 

A solution is proposed and implemented as a recurrent neural network that builds on a prominent
model for immediate serial recall (Botvinick & Plaut, 2006). The proposed dual model consists of 
two components – the neural network for serial recall, and a neural network that implements a 
mechanism for binding tokens and types. A set of simulations (including the List length effect, 
Serial recall curve, Word-frequency effect, Bigram frequency effect, and Recalling items on 
untrained positions) demonstrates that the new model solves the binding problem, while still being
able to reproduce the key findings in the field of immediate serial recall.
