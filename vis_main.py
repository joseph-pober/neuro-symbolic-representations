import numpy as np
from matplotlib import pyplot as plt
import vis

import directories as DIR

plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')

# data = np.load(DIR.histories_dir+"pae_code32.npy",allow_pickle=True)
# data_shape = np.shape(data[0].history['loss'])
# data_length = len(data)
#
# avg = {"loss":np.zeros(data_shape), "val_loss":np.zeros(data_shape)}
# # avg = np.average(data)
#
#
# for d in data:
# 	l = np.array(d.history['loss'])
# 	v_l = np.array(d.history['val_loss'])
# 	plt.plot(l,alpha=0.3)
# 	plt.plot(v_l,alpha=0.3)
# 	avg['loss'] += l/data_length
# 	avg['val_loss'] += v_l/data_length
# plt.plot(avg['loss'])
# plt.plot(avg['val_loss'])
# plt.title('model loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.show()

alpha = 0.2
dirs = [
	DIR.histories_dir+"0 layers/pae_code32-0.npy",
	DIR.histories_dir+"0 layers/pae_code30-2.npy",
	DIR.histories_dir+"0 layers/pae_code22-10.npy",
	DIR.histories_dir+"0 layers/pae_code16-16.npy"
]
vis.plot_history(dirs,plt)


dirs = [
	DIR.histories_dir+"1 layer/pae_code32-0.npy",
	DIR.histories_dir+"1 layer/pae_code30-2.npy",
	DIR.histories_dir+"1 layer/pae_code22-10.npy",
	DIR.histories_dir+"1 layer/pae_code16-16.npy"
]
vis.plot_history(dirs,plt,color1="orange",color2="royalblue")


# for i in range(len(dirs)):
# 	vis.plot_history(dirs[i],plt,alpha=1-(i*.15))

# vis.plot_history(DIR.histories_dir+"0 layers/pae_code32-0.npy",plt,alpha)
# vis.plot_history(DIR.histories_dir+"0 layers/pae_code30-2.npy",plt,alpha)
# vis.plot_history(DIR.histories_dir+"0 layers/pae_code28-4.npy",plt,alpha)
# vis.plot_history(DIR.histories_dir+"0 layers/pae_code22-10.npy",plt,alpha)
# vis.plot_history(DIR.histories_dir+"0 layers/pae_code16-16.npy",plt,alpha)



# l,v,a = vis.load_history(DIR.histories_dir+"pae_code32.npy")
# plt.plot(l,alpha=alpha,color="navy")
# plt.plot(v,alpha=alpha,color="firebrick")
# plt.plot(a['loss'],color="navy",linestyle='--')
# plt.plot(a['val_loss'],color="firebrick",linestyle='--')
#
# l,v,a = vis.load_history(DIR.histories_dir+"pae_code30-2.npy")
# plt.plot(l,alpha=alpha,color="royalblue")
# plt.plot(v,alpha=alpha,color="darkorange")
# plt.plot(a['loss'],color="royalblue")
# plt.plot(a['val_loss'],color="darkorange")
#
# l,v,a = vis.load_history(DIR.histories_dir+"pae_code28-4.npy")
# plt.plot(l,alpha=alpha,color="blueviolet")
# plt.plot(v,alpha=alpha,color="gold")
# plt.plot(a['loss'],color="blueviolet")
# plt.plot(a['val_loss'],color="gold")
#
# l,v,a = vis.load_history(DIR.histories_dir+"pae_code22-10.npy")
# plt.plot(l,alpha=alpha,color="cyan")
# plt.plot(v,alpha=alpha,color="saddlebrown")
# plt.plot(a['loss'],color="cyan")
# plt.plot(a['val_loss'],color="saddlebrown")
#
# l,v,a = vis.load_history(DIR.histories_dir+"pae_code16-16.npy")
# plt.plot(l,alpha=alpha,color="blue")
# plt.plot(v,alpha=alpha,color="red")
# plt.plot(a['loss'],color="blue")
# plt.plot(a['val_loss'],color="red")


plt.show()