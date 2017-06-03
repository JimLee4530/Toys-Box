import numpy as np

# data I/O
data = open('input.txt','r').read()
chars = list(set(data))
data_size, vocab_size = len(data), len(chars)
char_to_ix = {ch:i for i,ch in enumerate(chars)}
ix_to_char = {i:ch for i,ch in enumerate(chars)}

# hyperparameters
hidden_size = 100 # size of hidden layer of neurons
seq_length = 25 # number of steps to unroll the RNN for
learning_rate = 1e-1

# model parameters
Wxh = np.random.randn(4*hidden_size, vocab_size)*0.01 # input to hidden
Whh = np.random.randn(4*hidden_size, hidden_size)*0.01 # hidden to hidden
Why = np.random.randn(vocab_size, hidden_size)*0.01 # hidden to output
bh = np.zeros((4*hidden_size, 1)) # hidden bias
by = np.zeros((vocab_size, 1)) # output bias

def sigmoid(x):
  """
  A numerically stable version of the logistic sigmoid function.
  """
  pos_mask = (x >= 0)
  neg_mask = (x < 0)
  z = np.zeros_like(x)
  z[pos_mask] = np.exp(-x[pos_mask])
  z[neg_mask] = np.exp(x[neg_mask])
  top = np.ones_like(x)
  top[neg_mask] = z[neg_mask]
  return top / (1 + z)

def lossFun(inputs, targets, prev_h,prev_c):
    """
    inputs,targets are both list of integers.
    hprev is Hx1 array of initial hidden state
    returns the loss, gradiens on model parameters, and last hidden state
    """
    xs, hs, ys, ps, cs = {}, {}, {}, {}, {}
    hs[-1] = np.copy(prev_h)
    cs[-1] = np.copy(prev_c)
    loss = 0
    H = 100
    #forward pass
    for t in xrange(len(inputs)):
        xs[t] = np.zeros((vocab_size, 1)) # encode in 1-of-k representation
        xs[t][inputs[t]] = 1
        a = np.dot(Wxh, xs[t]) + np.dot(Whh,hs[t-1]) + bh #(N,4H)
        # print a
        ai,af,ao,ag = a[0:H], a[H:2*H], a[2*H:3*H], a[3*H:4*H]
        i = sigmoid(ai)
        f = sigmoid(af)
        o = sigmoid(ao)
        g = np.tanh(ag)
        cs[t] = f * cs[t-1] + i * g
        hs[t] = o * np.tanh(cs[t])
        ys[t] = np.dot(Why, hs[t]) + by # unnormalized log probabilities for next chars
        ps[t] = np.exp(ys[t]) / np.sum(np.exp(ys[t])) # probabilities for next chars
        loss += -np.log(ps[t][targets[t],0]) # softmax (cross - entropy loss)

    # backward pass:compute gradients going backwwards
    dWxh, dWhh, dWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)
    dbh, dby = np.zeros_like(bh), np.zeros_like(by)
    dnext_h = np.zeros_like(hs[0])
    dnext_c = np.zeros_like(cs[0])
    for t in reversed(xrange(len(inputs))):
        dy = np.copy(ps[t])
        dy[targets[t]] -= 1 # backprop into yz
        dWhy += np.dot(dy, hs[t].T)
        dby += dy
        dh = np.dot(Why.T, dy) + dnext_h # backprop into h
        do = dnext_h * np.tanh(cs[t])
        dnext_c += dnext_h* o * (1 - (np.tanh(cs[t]))**2)
        df = dnext_c * cs[t-1]
        dprev_c = dnext_c * f
        di = dnext_c * g
        dg = dnext_c * i
        dai = di * (sigmoid(ai) * (1-sigmoid(ai)))
        daf = df * (sigmoid(af) * (1-sigmoid(af)))
        dao = do * (sigmoid(ao) * (1-sigmoid(ao)))
        dag = dg * (1 - np.tanh(ag)**2)
        da = np.hstack((dai.T,daf.T,dao.T,dag.T))
        dx = da.dot(Wxh)
        dWxh += np.dot(da.T,xs[t].T)
        dprev_h = np.dot(Whh.T,da.T)
        # print hs[t-1].shape
        # print da.shape
        dWhh += np.dot(da.T, hs[t-1].T)
        dbh += da.T
    for dparam in [dWxh, dWhh, dWhy, dbh, dby]:
        np.clip(dparam, -5, 5, out=dparam) # clip to mitigate exploding gradients
    return loss, dWxh, dWhh,dWhy, dbh, dby, hs[len(inputs) - 1], cs[len(inputs) - 1]

# sample(h, seed_ix,n):
def sample(h, c, seed_ix, n):
    """
    sample a sequence of integers from the model 
    h is memory state, seed_ix is seed letter for first time step
    """
    x = np.zeros((vocab_size, 1))
    x[seed_ix] = 1
    ixes = []
    H,z =  Wxh.shape
    H = H/4
    for t in xrange(n):
        # print x.shape
        # print Wxh.shape
        # print h.shape
        # print Whh.shape
        # print bh.shape
        a = np.dot(Wxh, x) + np.dot(Whh,h) + bh  #(N,4H)
        # print a.shape
        ai,af,ao,ag = a[0:H], a[H:2*H], a[2*H:3*H], a[3*H:4*H]
        i = sigmoid(ai)
        f = sigmoid(af)
        o = sigmoid(ao)
        g = np.tanh(ag)
        # print f.shape
        c = f * c + i * g
        h = o * np.tanh(c)
        y = np.dot(Why, h) + by
        p = np.exp(y) / np.sum(np.exp(y))
        ix = np.random.choice(range(vocab_size), p=p.ravel())
        x = np.zeros((vocab_size, 1))
        x[ix] = 1
        ixes.append(ix)
    return ixes


# main loop
n, p = 0, 0
mWxh, mWhh, mWhy = np.zeros_like(Wxh), np.zeros_like(Whh),np.zeros_like(Why)
mbh, mby = np.zeros_like(bh),np.zeros_like(by) # memory variables for Adagrad
smooth_loss = -np.log(1.0/vocab_size)*seq_length # loss at iteration 0
while True:
    # perpare inputs (we're sweeping from left to right in steps seq_length long)
    if p+seq_length+1 >= len(data) or n==0:
        hprev = np.zeros((hidden_size, 1)) # reset RNN memory
        cprev = np.zeros((hidden_size, 1))
        p = 0 # go from start of data
    inputs = [char_to_ix[ch] for ch in data[p:p+seq_length]]
    targets = [char_to_ix[ch] for ch in data[p+1:p+seq_length+1]]

    # sample from the model now and then
    if n % 100 == 0:
        sample_ix = sample(hprev,cprev, inputs[0], 200)
        txt = ''.join(ix_to_char[ix] for ix in sample_ix)
        print '----\n %s \n----' %(txt, )

    # forward seq_length characters through the net and fetch gradient
    loss , dWxh, dWhh, dWhy, dbh, dby, hprev, cprev = lossFun(inputs, targets, hprev, cprev)
    smooth_loss = smooth_loss * 0.999 + loss * 0.001
    if n % 100 == 0: print 'iter %d, loss: %f' % (n, smooth_loss) #print progress

    #perform parameter update with Adagrad
    for param, dparam, mem in zip([Wxh,Whh,Why,bh,by],
                                  [dWxh,dWhh,dWhy,dbh,dby],
                                  [mWxh,mWhh,mWhy,mbh,mby]):
        mem += dparam * dparam
        param += -learning_rate * dparam / np.sqrt(mem + 1e-8)

    p += seq_length # move data pointer
    n += 1 # iteration counter