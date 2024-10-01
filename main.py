from torch.utils.data import DataLoader
from learner import Learner
from learner3 import Learner3
from learner4 import Learner4
from learner5 import Learner5
from learner6 import Learner6
from learner7 import *
from learner9 import *
from learner8 import Learner8
from loss import *
from dataset import *
import os
from sklearn import metrics
from sklearn.metrics import precision_recall_fscore_support
import argparse
from FFC import *
import warnings

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='PyTorch MIL Training')
parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
parser.add_argument('--w', default=0.0010000000474974513, type=float, help='weight_decay')
parser.add_argument('--modality', default='TWO', type=str, help='modality')
parser.add_argument('--input_dim', default=2048, type=int, help='input_dim')
parser.add_argument('--drop', default=0.6, type=float, help='dropout_rate')
parser.add_argument('--FFC', '-r', action='store_true',help='FFC')
args = parser.parse_args()

best_auc = 0
best_f1 = 0

normal_train_dataset = Normal_Loader(is_train=1, modality=args.modality)
normal_test_dataset = Normal_Loader(is_train=0, modality=args.modality)

anomaly_train_dataset = Anomaly_Loader(is_train=1, modality=args.modality)
anomaly_test_dataset = Anomaly_Loader(is_train=0, modality=args.modality)

normal_train_loader = DataLoader(normal_train_dataset, batch_size=30, shuffle=True,drop_last=True)
normal_test_loader = DataLoader(normal_test_dataset, batch_size=1, shuffle=True)

anomaly_train_loader = DataLoader(anomaly_train_dataset, batch_size=30, shuffle=True,drop_last=True) 
anomaly_test_loader = DataLoader(anomaly_test_dataset, batch_size=1, shuffle=True)

device = 'cpu'

# if args.FFC:
#     model = Learner2(input_dim=args.input_dim, drop_p=args.drop).to(device)
#     print("2")
# else:
#     model = Learner(input_dim=args.input_dim, drop_p=args.drop).to(device)
#     print("1")


# model = Learner2(input_dim=args.input_dim, drop_p=args.drop).to(device)
model = Learner8(input_dim=args.input_dim, drop_p=args.drop).to(device)
# model = Learner(input_dim=args.input_dim, drop_p=args.drop).to(device)


optimizer = torch.optim.Adam(model.parameters(), lr= args.lr, weight_decay=args.w)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[25, 50])
criterion = MIL

def train(epoch):
    print('\nEpoch: %d' % epoch)
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (normal_inputs, anomaly_inputs) in enumerate(zip(normal_train_loader, anomaly_train_loader)):
        inputs = torch.cat([anomaly_inputs, normal_inputs], dim=1)
        batch_size = inputs.shape[0]
        inputs = inputs.view(-1, inputs.size(-1)).to(device)
        outputs = model(inputs)
        loss = criterion(outputs, batch_size)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    print('loss = {}', train_loss/len(normal_train_loader))
    scheduler.step()

def test_abnormal(epoch):
    model.eval()
    global best_auc, best_f1
    auc = 0
    precision = 0
    recall = 0
    f1 = 0
    with torch.no_grad():
        for i, (data, data2) in enumerate(zip(anomaly_test_loader, normal_test_loader)):
            inputs, gts, frames = data
            inputs = inputs.view(-1, inputs.size(-1)).to(torch.device('cpu'))
            score = model(inputs)
            score = score.cpu().detach().numpy()
            score_list = np.zeros(frames[0])
            # step = np.round(np.linspace(0, frames[0]//16, 33))
            step = np.round(np.linspace(0, np.floor_divide(frames[0], 16), 33))

            for j in range(32):
                score_list[int(step[j])*16:(int(step[j+1]))*16] = score[j]

            gt_list = np.zeros(frames[0])
            for k in range(np.floor_divide(len(gts), 2)):
                s = gts[k*2]
                e = min(gts[k*2+1], frames)
                gt_list[s-1:e] = 1

            inputs2, gts2, frames2 = data2
            inputs2 = inputs2.view(-1, inputs2.size(-1)).to(torch.device('cpu'))
            score2 = model(inputs2)
            score2 = score2.cpu().detach().numpy()
            score_list2 = np.zeros(frames2[0])
            step2 = np.round(np.linspace(0, np.floor_divide(frames2[0], 16), 33))
            for kk in range(32):
                score_list2[int(step2[kk])*16:(int(step2[kk+1]))*16] = score2[kk]
            gt_list2 = np.zeros(frames2[0])
            score_list3 = np.concatenate((score_list, score_list2), axis=0)
            gt_list3 = np.concatenate((gt_list, gt_list2), axis=0)

            fpr, tpr, thresholds = metrics.roc_curve(gt_list3, score_list3, pos_label=1)
            auc += metrics.auc(fpr, tpr)

            pred_labels = (score_list3 > 0.5)  # Choose an appropriate threshold
            true_labels = gt_list3
            prec, rec, f1_score, _ = precision_recall_fscore_support(true_labels, pred_labels, average='binary',zero_division=1)

            precision += prec
            recall += rec
            f1 += f1_score


        avg_auc = auc / 56
        avg_precision = precision / 56
        avg_recall = recall / 56
        avg_f1 = f1 / 56
        
        
        print('AUC: {:.4f}'.format(avg_auc))
        print('Precision: {:.4f}'.format(avg_precision))
        print('Recall: {:.4f}'.format(avg_recall))
        print('F1-score: {:.4f}'.format(avg_f1))

        if best_auc < avg_auc:
            print('Saving...')
            print(' ')
            state = {
                'net': model.state_dict(),
                'best_auc': avg_auc,
                'best_f1': avg_f1,
                'best_precision': avg_precision,
                'best_recall': avg_recall
            }
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            torch.save(state, './checkpoint/ckpt.pth')
            best_auc = avg_auc

for epoch in range(0, 75):
    train(epoch)
    test_abnormal(epoch)