import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

datadir = './data/Demo'

package_label = pd.read_csv(os.path.join(datadir, 'package_label.tsv'), delimiter='\t',
                            names=['package', 'class', 'subclass'], index_col='package')


package_label['category'] = package_label['class'].map(str) + '-' + package_label['subclass'].map(str)
package_label['cls_label'] = LabelEncoder().fit_transform(package_label['class'])
package_label['label'] = LabelEncoder().fit_transform(package_label['category'])
package_label.to_csv('package_label_encoded.tsv', sep='\t', columns=['cls_label', 'label'], header=False)
