"""
Règles du jeux:
  - Amical, constructive, fun, esprit ouvert, nuance
  - Individuel: chacun m'envoie son rendu par email

Livraison attendu:
  - Un petit texte (format markdown de préférence): 
    - Vos commentaires sur le code initial
    - Les factorisations que vous avez réalisées
    - Autres (des doutes, des refacto que vous n'avez pas pu faire, des points à améliorer encore, vos conseils pour l'auteur du code, etc)
  - Codes refactorés (dans un fichier ou dossier ou autre)

On organisera ensuite un point de discussion
"""

from sklearn import metrics
import numpy as np
import pandas as pd

NA = -1

def best_threshold(y, p):
    scores = pd.Series({thr: metrics.f1_score(y, p > thr) for thr in np.arange(0, 1, 0.05)})
    if scores.max()>0.1:
        return scores.idxmax()
    else:
        # Si les f1-scores sont tous inferieurs a 0.1, on s'abstient de predire pour la categorie.
        return 1.
    return 


def recall_for_precision(y, p, precision=0.9):
    for thr in np.arange(0, 1, 0.05):
        if metrics.precision_score(y, p > thr,  zero_division=0) >= precision:
            return metrics.recall_score(y, p > thr)
    return 0


def logloss_with_na(target, pred, **kwargs):
    pred = pred[target.columns].values.ravel() # 2D => 1D
    target = target.values.ravel() # 2D => 1D
    notna = (target != NA)
    pred, target = pred[notna], target[notna]
    return metrics.log_loss(target, pred)

def get_boolean_metrics(Y_target, Y_pred_bool):
    """Retourne un panel de metriques a partir de predictions booleennes.
    
    Arguments:
        Y_target {pd.DataFrame} -- Valeur de verite pour chaque classe (0 ou 1).
        Y_pred_bool {pd.DataFrame} -- Prediction pour chaque classe (0 ou 1).
    
    Returns:
        dict -- metriques calculees.
    """
    boolean_metrics = {}
    boolean_metrics["moy_labels_par_mail"] = Y_pred_bool.sum(axis=1).mean()
    boolean_metrics["accuracy_totale_prop"] = (pd.Series(Y_pred_bool.values.tolist()) == pd.Series(Y_target.values.tolist())).mean()
    boolean_metrics["accuracy_totale_nombre"] = (pd.Series(Y_pred_bool.values.tolist()) == pd.Series(Y_target.values.tolist())).sum()

    no_labs = (Y_pred_bool.sum(axis=1) == 0)
    min_one_lab = (Y_pred_bool.sum(axis=1) > 0)
    mono_lab = (Y_pred_bool.sum(axis=1) == 1)
    multi_lab = (Y_pred_bool.sum(axis=1) > 1)

    boolean_metrics["sans_labels_prop"] = no_labs.mean()
    boolean_metrics["sans_labels_nombre"] = no_labs.sum()
    boolean_metrics["avec_labels_nombre"] = min_one_lab.sum()
    boolean_metrics["mono_labels_nombre"] = mono_lab.sum()
    boolean_metrics["multi_labels_nombre"] = multi_lab.sum()

    boolean_metrics["accuracy_sur_predits_prop"] = (pd.Series(Y_pred_bool[min_one_lab].values.tolist()) == pd.Series(Y_target[min_one_lab].values.tolist())).mean()
    boolean_metrics["accuracy_sur_predits_nombre"] = (pd.Series(Y_pred_bool[min_one_lab].values.tolist()) == pd.Series(Y_target[min_one_lab].values.tolist())).sum()
    boolean_metrics["accuracy_sur_monolabels_prop"] = (pd.Series(Y_pred_bool[mono_lab].values.tolist()) == pd.Series(Y_target[mono_lab].values.tolist())).mean()
    boolean_metrics["accuracy_sur_monolabels_nombre"] = (pd.Series(Y_pred_bool[mono_lab].values.tolist()) == pd.Series(Y_target[mono_lab].values.tolist())).sum()
    boolean_metrics["accuracy_sur_multilabels_prop"] = (pd.Series(Y_pred_bool[multi_lab].values.tolist()) == pd.Series(Y_target[multi_lab].values.tolist())).mean()
    boolean_metrics["accuracy_sur_multilabels_nombre"] = (pd.Series(Y_pred_bool[multi_lab].values.tolist()) == pd.Series(Y_target[multi_lab].values.tolist())).sum()
    
    return boolean_metrics


def get_all_metrics(Y_target, Y_pred_proba, in_thresholds={}):
    """Retourne un panel de metriques a partir de scores predits.
    
    Arguments:
        Y_target {pd.DataFrame} -- Scores entre 0 et 1 pour chaque classe.
            Peut eventuellement comprendre une colonne "ID" qui sera ignoree.
        Y_pred_proba {pd.DataFrame} -- Valeur de verite pour chaque classe.
            0 : non, 1 : oui, -1 : je ne sais pas. 
            Les eventuels -1 seront ignores dans l'evaluation.
    
    Keyword Arguments:
        in_thresholds {dict} -- {col : threshold} (default: {{}})
            Par defaut un dictionnaire vide, et les seuils seront calcules automatiquement. 
    
    Returns:
        tuple : (global_metrics, class_metrics, thresholds) avec : 
            thresholds la liste des seuils utilises (passes en argument ou calcules).
    """
    # On filtre la colonne ID pour ne garder que les colonnes de prediction.
    col_pred_proba = [c for c in Y_pred_proba.columns if c !='ID']
    pred_proba = Y_pred_proba[col_pred_proba].copy()
    # De meme pour la cible
    cols_target = list([c for c in Y_target.columns if c !='ID'])

    thresholds = in_thresholds.copy()

    class_metrics = []
    preds_bool = {}

    for col in cols_target:
        # On enleve du true et du preds les lignes "je ne sais pas" dans le true.
        y, p = Y_target[col], Y_pred_proba[col]
        y, p = y[y != NA], p[y != NA]

        # Choix du seuil : par defaut celui donne en entree, et sinon celui qui maximise le f1-score.
        if not bool(in_thresholds):
            thresholds[col] = best_threshold(y, p)
        thr = thresholds[col]
    
        preds_col = p.map(lambda x : 1 if x>thr else 0)
        preds_bool[col] = preds_col

        if y.sum() < 10:
            # S'il y a moins de 10 instances d'une classe dans le true, on ne calcule pas de metriques dessus.
            class_metrics.append({
                'Target': col,
                'Auc': None,
                'Logloss': None,
                'F1': None,
                'Precision': None,
                'Recall': None,
                'Recall_90': None,
                'Recall_80': None,
                'Threshold': thr,
                'Count': y.sum()
            })
        else:
            # Calcul des indicateurs par classe :
            class_metrics.append({
                'Target': col,
                'Auc': metrics.roc_auc_score(y, p),
                'Logloss': metrics.log_loss(y, p),
                'F1': metrics.f1_score(y, p > thr),
                'Precision': metrics.precision_score(y, p > thr, zero_division=0),
                'Recall': metrics.recall_score(y, p > thr),
                'Recall_90': recall_for_precision(y, p, 0.9),
                'Recall_80': recall_for_precision(y, p, 0.8),
                'Threshold': thr,
                'Count': y.sum()
            })

    # Agregation en dataframe et arrondis pour plus de lisibilite.
    class_metrics = pd.DataFrame(class_metrics).sort_values(by='F1', ascending=False)
    class_metrics[class_metrics.columns[1:]] = class_metrics[class_metrics.columns[1:]].applymap(lambda x: np.round(x,2))
    # Calcul des indicateurs globaux

    proba_metrics = {}
    proba_metrics["logloss"] = logloss_with_na(Y_target[cols_target], pred_proba)  
    proba_metrics["f1_score_global_pondere"] =(class_metrics.F1 * class_metrics.Count).sum()/class_metrics.Count.sum()

    Y_preds_bool = pd.DataFrame(preds_bool)
    bool_metrics = get_boolean_metrics(Y_target, Y_preds_bool)

    # Agregation sous forme de dataframe a multiindex pour plus de lisibilite.
    global_metrics = {"proba_metrics": proba_metrics, "bool_metrics": bool_metrics}
    global_metrics =  {(outerKey, innerKey): values for outerKey, innerDict in global_metrics.items() for innerKey, values in innerDict.items()}
    global_metrics = pd.DataFrame(global_metrics, index=["stat"]).T
    global_metrics.stat = global_metrics.stat.map(lambda x : np.round(x,2))

    return global_metrics, class_metrics, thresholds
