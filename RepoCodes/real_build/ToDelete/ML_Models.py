
#################################################################
#                      MACHINE LEARNING                         #
#################################################################
class EarlyStopping():
    '''
    Class to determine if stopping criteria met in back propogation
    '''
    def __init__(self,model, patience = 50, min_delta = 1e-6, restore_best_weights = True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_model     = None
        self.best_loss  = None
        self.counter    = 0
        self.status     = ""
        
    def __call__(self, model, val_loss):
        if self.best_loss == None:
            self.best_loss = val_loss
            self.best_model = copy.deepcopy(model)
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_model.load_state_dict(model.state_dict())
        elif self.best_loss - val_loss < self.min_delta:
            self.counter += 1
        if self.counter >= self.patience:
            self.status = f"Stopped on {self.counter}"
            if self.restore_best_weights:
                model.load_state_dict(self.best_model.state_dict())
            return True
        self.status = f"{self.counter}/{self.patience}"
        return False
    
    
def trainingloop(model,optimizer,x_train,y_train,x_test,y_test,dataloader_train, 
                    fair = True, patience = 25, min_delta = 1e-6, restore_best_weights = True):
        '''
        Function to excecute backpropagation with early stopping
        '''
        es          = EarlyStopping(model, patience = patience, min_delta = min_delta, 
                                    restore_best_weights = restore_best_weights)
        epoch       = 0
        done        = False
        history,  historyt   = [], []
        while epoch<300 and not done:
            epoch   += 1
            steps   = list(enumerate(dataloader_train))
            pbar    = tqdm.tqdm(steps)
            model.train()
            for i, (x_batch, y_batch) in pbar:
                y_batch_pred    = model(x_batch.to(device)).flatten()
                loss            = loss_fn(y_batch_pred , y_batch.flatten().to(device))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss, current   = loss.item(), (i + 1)* len(x_batch)
                if i == len(steps)-1:
                    model.eval()
                    if not fair:
                        pred    = model(x_train).flatten()
                        vloss   = loss_fn(pred, y_train.flatten())
                    else:
                        pred    = model(x_test).flatten()
                        vloss   = loss_fn(pred, y_test.flatten())
                    history.append(float(vloss))
                    pred    = model(x_train).flatten()
                    tloss   = loss_fn(pred, y_train.flatten())
                    historyt.append(float(tloss))
                    if es(model,vloss): done = True
                    pbar.set_description(f"Epoch: {epoch}, tloss: {loss}, vloss: {vloss:>7f}, EStop:[{es.status}]")
                else:
                    pbar.set_description(f"Epoch: {epoch}, tloss {loss:}")
            y_pred = model(x_train).flatten()
            mse = loss_fn(y_pred,y_train.flatten())
        return model, history, historyt
        # scheduler1.step()
    
class NetTanh(nn.Module):
    def __init__(self, D_in, H, D, D_out):
        """
        In the constructor, instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(NetTanh, self).__init__()
        self.inputlayer         = nn.Linear(D_in, H)
        self.middle             = nn.Linear(H, H)
        self.lasthiddenlayer    = nn.Linear(H, D)
        self.outputlayer        = nn.Linear(D, D_out)
    def forward(self, x):
        """
        In the forward function, accept a variable of input data and return
        a variable of output data. Use modules defined in the constructor, as
        well as arbitrary operators on variables.
        """
        y_pred = self.outputlayer(self.PHI(x))
        return y_pred
        
    def PHI(self, x):
        h_relu = self.inputlayer(x).tanh()
        for i in range(3):
            h_relu = self.middle(h_relu).tanh()
            # x = F.relu(x)
        phi = self.lasthiddenlayer(h_relu)
        return phi
    

class Net(nn.Module):
    """
    Define a simple fully connected model
    """
    def __init__(self, in_count, out_count, First = 128 , deep  = 32, Ndeep = 4, isrelu = True):
        super(Net,self).__init__()
        d2 = deep
        self.Ndeep  = Ndeep
        self.fc1    = nn.Linear(in_count,First,bias = True)
        self.fc2    = nn.Linear(First,d2,bias = True)
        self.fc3    = nn.Linear(d2,d2,bias = True)
        self.fc4    = nn.Linear(d2,d2,bias = True)
        self.do     = nn.Dropout(.25)
        self.fcend  = nn.Linear(d2,out_count)
        self.tanh   = nn.Tanh()
        if isrelu:
            self.relu   = nn.ReLU()
        else:
            self.relu   = nn.Tanh()
            
        self.seq    = nn.Sequential(
                    nn.Linear(in_count, 24),
                    nn.ReLU(),
                    nn.Linear(24, 12),
                    nn.ReLU(),
                    nn.Linear(12, 6),
                    nn.ReLU(),
                    nn.Linear(6, 1)
                )
    def forward(self, x):
        # x = self.seq(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        for i in range(self.Ndeep):
            x = self.relu(self.fc4(x)) 
        return self.fcend(x)
    
class Netc(nn.Module):
    """
    Define a simple 1D CNN
    """
    def __init__(self):
        super(Netc, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 1)
    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.linear(x)
    
class Simple1DCNN(torch.nn.Module):
    """
    Define a simple 1D CNN
    """
    def __init__(self):
        super(Simple1DCNN, self).__init__()
        self.fc1 = torch.nn.Linear(3,7)
        self.layer1 = torch.nn.Conv1d(in_channels=32, out_channels=20, kernel_size=5, stride=2)
        self.act1 = torch.nn.ReLU()
        self.layer2 = torch.nn.Conv1d(in_channels=20, out_channels=10, kernel_size=1)
        self.fc2 = torch.nn.Linear(20,1)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.layer1(x)
        x = self.act1(x)
        x = self.layer2(x)
        x = self.fc2(x)
        return F.limear(x)