import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import logging

import pandas as pd

#### KEEP THE GENDER DATA ON TOP LEVEL

def analyse_classification(y_pred, x_val, y_val, gender_val):
    df = pd.DataFrame(x_val)
    df = pd.concat([df, pd.DataFrame({"prediction": y_pred > 0, "actual": list(map(bool, y_val)), "gender": gender_val})],axis=1)
    winners = df[y_pred> 0]
    print(winners.groupby("gender").mean().iloc[:,:2].sum(axis= 1))
    df["TP"] = df["prediction"] & df["actual"]
    df["TN"] = ~df["prediction"] & ~df["actual"]
    df["FP"] = df["prediction"] & ~df["actual"]
    df["FN"] = ~df["prediction"] & df["actual"]
    agg = df.groupby("gender").sum()
    print(agg)



# now for validating this approach we create the same data but this time balanced
def eval(x_val, y_val, gender_val):
    y_pred = net(x_val).detach().numpy()
    print("accuracy: ", ((y_pred > 0) == y_val.detach().numpy()).sum().item() / len(y_val))
    analyse_classification(y_pred, x_val, y_val, gender_val)

def create_data(n_data=100000, unfairness=0, only_male=False, only_female=False, change_latent_gender_only_female=False, change_latent_gender_only_male=False):
    print("Generate data...")
    # generate random data
    x = np.random.randn(n_data, 2)
    # sample 90% male and 10% female
    gender = np.random.choice([0, 1], size=n_data, p=[0.5, 0.5]).astype(float)
    gender_for_model = gender

    if only_male:
        gender_for_model = np.ones(n_data)

    if only_female:
        gender_for_model = np.zeros(n_data)

    assert not (only_male and only_female)
    # convert gender to float

    # concatenate gender to the data
    # think of x without the gender as scores in a test and then add unfairness with the gender variable.
    y = np.array([1 if sum(x[i,:]) > cutoff - gender[i]* unfairness else 0 for i in range(len(x))])

    latent_gender = gender
    if change_latent_gender_only_female:
        latent_gender = np.zeros(n_data)

    if change_latent_gender_only_male:
        latent_gender = np.ones(n_data)

    x = np.concatenate((x, latent_gender.reshape(-1, 1)), axis=1)  # concatenate latent information, we cant manipulate this, this is in the data

    x = np.concatenate((x, gender_for_model.reshape(-1, 1)), axis=1) # concatenate gender information that we can manipulate

    # undersample the majority class
    mask_accepted = y == 1
    mask_male = gender == 1



    mask_accepted_and_male = mask_accepted & mask_male
    mask_accepted_and_female = mask_accepted & ~mask_male
    mask_not_accepted_and_male = ~mask_accepted & mask_male
    mask_not_accepted_and_female = ~mask_accepted & ~mask_male
    minimal_data = 1000 # very ugly and hacky
    # minimal_data = min(mask_accepted_and_male.sum(),
    #                   mask_accepted_and_female.sum(),
    #                   mask_not_accepted_and_male.sum(),
    #                   mask_not_accepted_and_female.sum())

    def get_balanced_data(data):
        return np.concatenate([data[mask_accepted_and_male][:minimal_data],
                               data[mask_accepted_and_female][:minimal_data],
                               data[mask_not_accepted_and_male][:minimal_data],
                               data[mask_not_accepted_and_female][:minimal_data]])
    # balance data
    x = get_balanced_data(x)
    y = get_balanced_data(y)
    gender = get_balanced_data(gender)


    # get a permutation of the data
    perm = np.random.permutation(4*minimal_data)
    x = x[perm]
    y = y[perm]
    gender = gender[perm]

    print(y.sum() / (4*minimal_data) * 100, "% class 2 samples")

    # convert x and y to tensors
    x = torch.from_numpy(x).float()
    y = torch.from_numpy(y).float()
    return x, y, gender


# set logging level
#logging.basicConfig(level=logging.DEBUG)

# create data with 2 classes 1000 samples and 2 features using numpy

cutoff = 2
# set numpy seed
for use_bias, bake_in_bias in [(True, True)]:#, (False, True)]:
    print(100*"-")
    print(f"bake_in_bias: {bake_in_bias}, use_bias: {use_bias}")
    np.random.seed(1)


    x,y, gender = create_data(unfairness=1)
    n = len(x)
    # split train test data
    x_train = x[:int(n*0.8)]
    y_train = y[:int(n*0.8)]
    x_test = x[int(n*0.8):]
    y_test = y[int(n*0.8):]
    gender_train = gender[:int(n*0.8)]
    gender_test = gender[int(n*0.8):]

    # create a neural network where the last variable jumps to the last layer

    n_attributes = x.shape[1]
    # create a neural network
    class Net(nn.Module):
        def __init__(self, erased_last_var=False):
            super(Net, self).__init__()
            self.erased_last_var = erased_last_var


            self.fc1 = nn.Linear(n_attributes  - int(erased_last_var) , 10)
            self.fc2 = nn.Linear(10, 5)
            self.fc3 = nn.Linear(5, 1)
            self.last_var_detached = False
            self.rest_detached = False

        def detach_last_var(self): # so we can freeze the weights of the last variable
            self.last_var_detached = True

        def attach_last_var(self): # so we can unfreeze the weights of the last variable
            self.last_var_detached = False

        def detach_all_except_last_var(self):
            self.rest_detached = True

        def attach_all_except_last_var(self):
            self.rest_detached = False

        def forward(self, x):
            # cut off last variable
            x_last = x[:, -1]
            x_rest = x[:, :-1]

            if not self.erased_last_var:
                # right before the last layer we add the last variable, to make it have a strong effect on the output
                x = x
            else:
                x = x_rest

            x = self.fc1(x)
            x = self.fc2(x)
            x = self.fc3(x)

            # remove last dimension
            x = x.reshape(-1)
            return x

    net = Net(erased_last_var=(not use_bias))

    # create optimizer use adam
    optimizer = optim.Adam(net.parameters(), lr=0.001)

    if bake_in_bias:
        # train with information only in the last variable to prevent net from learning latent variable
        # permute rows
        net.detach_all_except_last_var()
        for epoch in range(3000):
            #if epoch % 1000 == 0:
            #    print(epoch)
            #    optimizer = optim.Adam(net.parameters(), lr=0.001*0.9**(epoch/1000))
            perm = np.random.permutation(len(x_train[:, :-1]))
            x_perm = x_train[:, :-1]#[perm] # TODO
            x_train_no_info = torch.cat((x_perm, x_train[:, -1].reshape(-1,1)), 1)

            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            out = net(x_train_no_info)
            loss = F.binary_cross_entropy_with_logits(out, y_train)
            loss.backward()

            # set gradients to zero so that only that last variable is trained:
            for n, w in net.named_parameters():
                if n == "fc1.weight":
                    w.grad[:,:-1] = 0
                if n == "fc1.bias":
                    w.grad[:-1] = 0

            optimizer.step()

        net.attach_all_except_last_var()
        # show metrics for training, should have learned the naive model
        x_test_no_info = x_test # torch.cat((torch.zeros(x_test[:, :-1].shape), x_test[:, -1].reshape(-1,1)), 1)
        outputs = net(x_test_no_info)
        print("Check: accuracy: ", max(((outputs > 0) == y_test).sum().item()/len(y_test), ((outputs < 0) == y_test).sum().item()/len(y_test)), "Should be about ", 1-y_test.sum().item()/len(y_test)) # TODO not sure why it sometimes inverses the classes, but this is a temporary fix
        analyse_classification(outputs.detach().numpy(), x_test, y_test, gender_test)
        # then remove the last variable from training
        #net.detach_last_var() # TODO

    # keep training the network
    for epoch in range(5000):

        optimizer.zero_grad()
        out = net(x_train)
        loss = F.binary_cross_entropy_with_logits(out, y_train)
        loss.backward()

        # practically detach the last variable
        #if bake_in_bias:
        #    for n, w in net.named_parameters():
        #        if n == "fc2.weight":
        #            w.grad[:, -1] = 0
        optimizer.step()


        y_pred = net(x_test).detach().numpy()
        if epoch % 100 == 0:
            with torch.no_grad():
                print(50*"*")
                x_val, y_val, gender_val = create_data(unfairness=0, only_female=True)
                eval(x_val, y_val, gender_val)
                #x_val, y_val, gender_val = create_data(unfairness=1, change_latent_gender_only_female=True)
                #eval(x_val, y_val, gender_val)



    # The baked model should respond to the change in the last variable. Otherwise the other variable has learned the latent variable


    print(10*"*")
    print("all male")
    print("fair:")
    x_val_fair, y_val_fair, gender_val_fair = create_data(unfairness=0, only_male=True)
    eval(x_val_fair, y_val_fair, gender_val_fair)
    print(10*"*")
    print("all female")
    print("fair:")

    x_val_fair, y_val_fair, gender_val_fair = create_data(unfairness=0, only_female=True)
    eval(x_val_fair, y_val_fair, gender_val_fair)
    x_val_fair_male = x_val_fair.clone()
    x_val_fair_male[:,-1] = 1
    eval(x_val_fair_male, y_val_fair, gender_val_fair)
    x_val_fair_female = x_val_fair.clone()
    # combine male and female predictions

    y_pred_female = net(x_val_fair_female).detach().numpy()
    y_pred_male = net(x_val_fair_male).detach().numpy()
    merged_pred = y_pred_female + y_pred_male
    print("accuracy: ", ((merged_pred > 0) == y_val_fair.detach().numpy()).sum().item() / len(y_val_fair))
    analyse_classification(merged_pred, x_val_fair, y_val_fair, gender_val_fair)