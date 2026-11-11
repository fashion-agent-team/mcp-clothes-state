import csv, hashlib, numpy as np, time, sys, collections
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
CLASSES = ['화이트','블랙','베이지','핑크','그레이','스카이블루','네이비','브라운','블루','카키','레드','그린','라벤더','실버','옐로우','와인','민트','퍼플','오렌지','골드']
rows = list(csv.reader(open('crops/meta.csv')))
X = np.fromfile('feats.f32', dtype='<f4').reshape(len(rows), -1)
def split(img):
    h = int(hashlib.md5(img.encode()).hexdigest()[:8], 16) % 100
    return 'test' if h < 15 else 'val' if h < 25 else 'train'
sp = np.array([split(r[1]) for r in rows]); y = np.array([CLASSES.index(r[4]) if r[4] in CLASSES else -1 for r in rows])
ok = y >= 0
tr = ok & (sp=='train'); va = ok & (sp=='val'); te = ok & (sp=='test')
print(tr.sum(), va.sum(), te.sum())
sc = StandardScaler().fit(X[tr])
Xs = sc.transform(X).astype(np.float32)
t=time.time()
m = MLPClassifier(hidden_layer_sizes=(256,128), alpha=1e-3, batch_size=512, learning_rate_init=1e-3, max_iter=60,
                  early_stopping=True, validation_fraction=0.08, n_iter_no_change=6, random_state=0, verbose=False)
m.fit(Xs[tr], y[tr])
print('train time', round(time.time()-t), 'iters', m.n_iter_)
for name, msk in (('val', va), ('test', te)):
    p = m.predict(Xs[msk]); print(name, '20색 정확도', (p==y[msk]).mean().round(3))
np.save('pred_test.npy', np.stack([np.where(te)[0], m.predict(Xs[te]), m.predict_proba(Xs[te]).max(1)*1000], 1))
import pickle; pickle.dump((sc, m), open('mlp.pkl','wb'))
