# Paper 01 — Differentially Private Federated Learning: A Systematic Review

## 1. Citation

Title: Differentially Private Federated Learning: A Systematic Review

Authors: Jie Fu, Yuan Hong, Xinpeng Ling, Leixia Wang, Xun Ran, Zhiyu Sun, Wendy Hui Wang, Zhili Chen, and Yang Cao

Year:2025

Venue / Source: ACM



## 2. Why am I reading this?
To build my overall DP-FL taxonomy, understand client/example level dp, and basic understanding.



## 3. Federated Learning

### What is Federated Learning?
It is distributed machine learning where model trains locally on client devices and the updates which are necessarily the weights and gradients get aggregated on a server.

### What is a client?
A client is a participating device or entity on the FL system

### What remains local?
The data remains local, never leaves the device.

### What information is sent to the server?
 gradients, parameters

### What does the server do?
server aggragates them, either via FedAvg or FedProx etc.

## 4. Why FL alone is not private

What information can still leak?
The gradients can still leak private info, backdoor attack, inference attack can be performed and the data can be reconstructed from the gradients.

Why is Differential Privacy needed?
Differential Privacy is used to ensure the gradients doesn't leak the data, it can be used by adding a small amount of noise so that particular data can be hidden.

## 5. Differential Privacy

### What does (epsilon, delta)-DP mean intuitively?
achieving epsilon privacy with a delta probablity of failure

### What is a neighboring dataset?
dataset after addition of one sample ( SL_DP ) or one client ( CL_DP )

## 6. Privacy Levels

### Example / Record-Level DP
Each data of every client is protected

### Client / User-Level DP
Each client is protected, their whole identity along with data

### Main Difference
It hides the data and client level dp hides the client's participation

### Which one is relevant to my journal work?
Client level dp.

## 7. DP Placement / Trust Model

### Central DP
Clients send update to trusted server, server adds noise

### Local DP
The server is not fully trusted so clients deals with noise addition themselves.

### Distributed DP
Instead of relying on one server, the noise addition and shuffling happens between many.

### Where is noise added?
on the aggregated update

### Who has to be trusted?
client

## 8. DP-FL Pipeline

Write the complete process in my own words:

1. Clients gets an initial model broadcasted by server
2. clients train the model locally using their own data
3. If LDP then adds noise before broadcasting
4. IF CDP then server collects all update and then adds noise
5. Server aggregates and updates global model
6. This process continues till model converges.


## 9. Important Parameters

### epsilon
Privacy measurement tool. The less the more privacy

### delta
Probabilty of privacy Failure

### clipping bound C
Parameter clipping bound

### noise multiplier sigma
The gaussian noise parameter that's added.

### sampling rate q
Number of clients participating rate in each round


## 10.My Summary

The paper explored and systematized the way DP can be applied on Various kind of FL, also their classifications and difference like SL_DP, CL_DP etc. They also discussed where currently FL is being used like on GNN and stuff, also where can we see the real life application like in healthcare, Iot, Finance etc. Lastly it discussed the problems and future direction, discusing 6 possible way,where I found the LLM aspect of it interesting.For my work, the most important takeaway is that I must define the protected object, adjacency relation, trust model, clipping mechanism, sampling mechanism, and privacy accountant precisely before claiming client-level DP