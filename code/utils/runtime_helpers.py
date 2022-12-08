import json
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.metrics import log_loss, classification_report

def eval_and_print_metrics(model, X_train, y_train, X_test, y_test):

    probs = model.predict_proba(X_test)
    preds = model.predict(X_test)

    acc = accuracy_score(
        y_test, 
        preds
    )
    roc_auc = roc_auc_score(
        y_test, 
        probs, 
        multi_class='ovo',
        average='weighted'
    )
    ll = log_loss(
        y_test, 
        probs,     
    )
    cr = classification_report(
        y_test,
        preds, 
        zero_division=0
    )

    print(f'Accuracy: {acc}')
    print(f'ROC AUC: {roc_auc}')
    print(f'Logloss: {ll}')
    print(cr)

    return probs, preds, acc, roc_auc, ll

#
# 
# #
def get_exp_var_ratio(pca_):
    agg = 0
    for v in pca_.explained_variance_ratio_:
        agg += v
    return agg

def get_exp_idxs(pca_):
    idxs = []
    for i in range(0, pca_.n_components):
        idxs.append(f'PC-{i+1}')
    return idxs
    
# 
# 
# 
def get_crop_dict():
    with open(
        '../../data/ref_agrifieldnet_competition_v1/ref_agrifieldnet_competition_v1_labels_train/ref_agrifieldnet_competition_v1_labels_train_001c1/ref_agrifieldnet_competition_v1_labels_train_001c1.json') as ll:
        label_json = json.load(ll)
    crop_dict = {asset.get('values')[0]:asset.get('summary') for asset in label_json['assets']['raster_labels']['file:values']}
    return crop_dict

def labeler(labeled, crop_dict):
    return np.array([crop_dict.get(f'{int(i)}') for i in labeled])

#
def get_features(agg_idxs, agg_metrics, ext=[]):
    '''
    
    Parameters
    ----------
        - agg_idxs
        - agg_metrics
        - ext

    '''   

    selected = []
    selected += ext
    for am in agg_metrics:
        selected += [b + am for b in agg_idxs]
    return selected

#
def get_crop_name_from_id(crop_list, c_id):        
    srch = [c for c in crop_list if c['id'] == c_id]
    ret_val = None

    if len(srch) != 0:
        ret_val = srch[0]['name']
    
    return ret_val

