import pandas as pd
df = pd.read_csv("classification_results.csv")

print(f"Total patients: {len(df)}")
print(f"Correct predictions: {(df.true_label == df.pred_label).sum()}")
print(f"Accuracy: {(df.true_label == df.pred_label).mean()*100:.2f}%")
print(f"\nLGG correct: {((df.true_label==0) & (df.pred_label==0)).sum()}")
print(f"HGG correct: {((df.true_label==1) & (df.pred_label==1)).sum()}")
print(f"\nMean Dice: {df.dice_mean.mean():.4f}")
print(f"\nWorst 5 patients:")
print(df.nsmallest(5, 'dice_mean')[['BraTS21ID','true_label','pred_label','prob_hgg','dice_mean']])