# -*- coding: utf-8 -*-

"""
This is the main all-in-one script for generating submissions.  Call it using

    python main.py

It will automatically train the selected model, create predictions, and
optimize the incentives.
"""

import sys
import os
import json
from datetime import datetime
from shutil import copyfile
import pandas as pd
from data import get_data
from renewal_predictor import get_model
from incentives import get_optimal_policy_incentives, total_net_revenue


# Settings
if len(sys.argv) > 1:
    use_model = sys.argv[1]
else:
    use_model = 'xgb'
cap_renewal_prob = True

# Load data
X_train, y_train, df_train, X_test, df_test = get_data()

# Load model
model, model_info = get_model(use_model, True)

# Premiums and predicted renewal probs
policy_premiums = df_test['premium'].values
policy_renewal_probs = model.predict_proba(X_test)[:, 1]

# Find optimal incentives
policy_incentives = get_optimal_policy_incentives(
    policy_renewal_probs,
    policy_premiums,
    cap_renewal_prob=cap_renewal_prob
)

# Reporting
print('\n\n################################################\n\n')
metrics = {}
# Part A
# w1 = 0.7
# metrics['part_a_score'] = w1 * model_info['metrics']['inner_cv_auc']
# print('Estimated Part A score: {}'.format(
#     metrics['part_a_score']
# ))
# Part B
metrics['total_net_revenue'] = total_net_revenue(policy_incentives,
                                                 policy_renewal_probs,
                                                 policy_premiums)
print("Total net revenue: {:,}".format(
    metrics['total_net_revenue']
))
print('\n\n################################################\n\n')


# Write to disk
submission = pd.DataFrame({
    'id': df_test['id'],
    'renewal': policy_renewal_probs,
    'incentives': policy_incentives
})
submissions_dir = './save/submission_%s' % datetime.now().isoformat()
os.mkdir(submissions_dir)
submission.to_csv(os.path.join(submissions_dir, 'submission.csv'), index=False)
with open(os.path.join(submissions_dir, 'metrics.json'), 'w') as f:
    json.dump(metrics, f)
# Copy model info for posterity
copyfile('./save/models/%s/info.json' % use_model,
         os.path.join(submissions_dir, 'model_info.json'))
print('\n\nWrote submissions dir %s' % submissions_dir)
