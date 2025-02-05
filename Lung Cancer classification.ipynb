{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "209fa180",
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "import pandas as pd\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.naive_bayes import GaussianNB\n",
    "from sklearn.metrics import accuracy_score\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "ee07c773",
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>GENDER</th>\n",
       "      <th>AGE</th>\n",
       "      <th>SMOKING</th>\n",
       "      <th>YELLOW_FINGERS</th>\n",
       "      <th>ANXIETY</th>\n",
       "      <th>PEER_PRESSURE</th>\n",
       "      <th>CHRONIC DISEASE</th>\n",
       "      <th>FATIGUE</th>\n",
       "      <th>ALLERGY</th>\n",
       "      <th>WHEEZING</th>\n",
       "      <th>ALCOHOL CONSUMING</th>\n",
       "      <th>COUGHING</th>\n",
       "      <th>SHORTNESS OF BREATH</th>\n",
       "      <th>SWALLOWING DIFFICULTY</th>\n",
       "      <th>CHEST PAIN</th>\n",
       "      <th>LUNG_CANCER</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>M</td>\n",
       "      <td>69</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "      <td>2</td>\n",
       "      <td>2</td>\n",
       "      <td>2</td>\n",
       "      <td>2</td>\n",
       "      <td>2</td>\n",
       "      <td>YES</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>M</td>\n",
       "      <td>74</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "      <td>2</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "      <td>2</td>\n",
       "      <td>2</td>\n",
       "      <td>YES</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>F</td>\n",
       "      <td>59</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "      <td>NO</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>M</td>\n",
       "      <td>63</td>\n",
       "      <td>2</td>\n",
       "      <td>2</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "      <td>2</td>\n",
       "      <td>NO</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>F</td>\n",
       "      <td>63</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>NO</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "  GENDER  AGE  SMOKING  YELLOW_FINGERS  ANXIETY  PEER_PRESSURE  \\\n",
       "0      M   69        1               2        2              1   \n",
       "1      M   74        2               1        1              1   \n",
       "2      F   59        1               1        1              2   \n",
       "3      M   63        2               2        2              1   \n",
       "4      F   63        1               2        1              1   \n",
       "\n",
       "   CHRONIC DISEASE  FATIGUE   ALLERGY   WHEEZING  ALCOHOL CONSUMING  COUGHING  \\\n",
       "0                1         2         1         2                  2         2   \n",
       "1                2         2         2         1                  1         1   \n",
       "2                1         2         1         2                  1         2   \n",
       "3                1         1         1         1                  2         1   \n",
       "4                1         1         1         2                  1         2   \n",
       "\n",
       "   SHORTNESS OF BREATH  SWALLOWING DIFFICULTY  CHEST PAIN LUNG_CANCER  \n",
       "0                    2                      2           2         YES  \n",
       "1                    2                      2           2         YES  \n",
       "2                    2                      1           2          NO  \n",
       "3                    1                      2           2          NO  \n",
       "4                    2                      1           1          NO  "
      ]
     },
     "execution_count": 2,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df = pd.read_csv(\"lungcancer.csv\")\n",
    "df.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "c7ea59a5",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\anike\\anaconda3\\Lib\\site-packages\\sklearn\\preprocessing\\_label.py:114: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().\n",
      "  y = column_or_1d(y, warn=True)\n"
     ]
    }
   ],
   "source": [
    "from sklearn.preprocessing import LabelEncoder\n",
    "encoder=LabelEncoder()\n",
    "df['GENDER']=encoder.fit_transform(df[['GENDER']])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "0777ee62",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\anike\\anaconda3\\Lib\\site-packages\\sklearn\\preprocessing\\_label.py:114: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().\n",
      "  y = column_or_1d(y, warn=True)\n"
     ]
    }
   ],
   "source": [
    "df['LUNG_CANCER']=encoder.fit_transform(df[['LUNG_CANCER']])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "adfbe379",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>GENDER</th>\n",
       "      <th>AGE</th>\n",
       "      <th>SMOKING</th>\n",
       "      <th>YELLOW_FINGERS</th>\n",
       "      <th>ANXIETY</th>\n",
       "      <th>PEER_PRESSURE</th>\n",
       "      <th>CHRONIC DISEASE</th>\n",
       "      <th>FATIGUE</th>\n",
       "      <th>ALLERGY</th>\n",
       "      <th>WHEEZING</th>\n",
       "      <th>ALCOHOL CONSUMING</th>\n",
       "      <th>COUGHING</th>\n",
       "      <th>SHORTNESS OF BREATH</th>\n",
       "      <th>SWALLOWING DIFFICULTY</th>\n",
       "      <th>CHEST PAIN</th>\n",
       "      <th>LUNG_CANCER</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>1</td>\n",
       "      <td>69</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "      <td>2</td>\n",
       "      <td>2</td>\n",
       "      <td>2</td>\n",
       "      <td>2</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>1</td>\n",
       "      <td>74</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "      <td>2</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "      <td>2</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>0</td>\n",
       "      <td>59</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>1</td>\n",
       "      <td>63</td>\n",
       "      <td>2</td>\n",
       "      <td>2</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "      <td>2</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>0</td>\n",
       "      <td>63</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   GENDER  AGE  SMOKING  YELLOW_FINGERS  ANXIETY  PEER_PRESSURE  \\\n",
       "0       1   69        1               2        2              1   \n",
       "1       1   74        2               1        1              1   \n",
       "2       0   59        1               1        1              2   \n",
       "3       1   63        2               2        2              1   \n",
       "4       0   63        1               2        1              1   \n",
       "\n",
       "   CHRONIC DISEASE  FATIGUE   ALLERGY   WHEEZING  ALCOHOL CONSUMING  COUGHING  \\\n",
       "0                1         2         1         2                  2         2   \n",
       "1                2         2         2         1                  1         1   \n",
       "2                1         2         1         2                  1         2   \n",
       "3                1         1         1         1                  2         1   \n",
       "4                1         1         1         2                  1         2   \n",
       "\n",
       "   SHORTNESS OF BREATH  SWALLOWING DIFFICULTY  CHEST PAIN  LUNG_CANCER  \n",
       "0                    2                      2           2            1  \n",
       "1                    2                      2           2            1  \n",
       "2                    2                      1           2            0  \n",
       "3                    1                      2           2            0  \n",
       "4                    2                      1           1            0  "
      ]
     },
     "execution_count": 5,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "7f64b7e2",
   "metadata": {},
   "outputs": [],
   "source": [
    "X=df.drop('LUNG_CANCER',axis=1)\n",
    "y=df['LUNG_CANCER']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "f9cc3073",
   "metadata": {},
   "outputs": [],
   "source": [
    "X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=0.2,random_state=42)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "775085bd",
   "metadata": {},
   "outputs": [],
   "source": [
    "model=GaussianNB()\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "9d95149f",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<style>#sk-container-id-1 {color: black;}#sk-container-id-1 pre{padding: 0;}#sk-container-id-1 div.sk-toggleable {background-color: white;}#sk-container-id-1 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-1 label.sk-toggleable__label-arrow:before {content: \"▸\";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-1 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-1 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-1 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-1 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-1 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-1 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: \"▾\";}#sk-container-id-1 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-1 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-1 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-1 div.sk-parallel-item::after {content: \"\";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-1 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-serial::before {content: \"\";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-1 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-1 div.sk-item {position: relative;z-index: 1;}#sk-container-id-1 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-1 div.sk-item::before, #sk-container-id-1 div.sk-parallel-item::before {content: \"\";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-1 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-1 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-1 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-1 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-1 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-1 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-1 div.sk-label-container {text-align: center;}#sk-container-id-1 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-1 div.sk-text-repr-fallback {display: none;}</style><div id=\"sk-container-id-1\" class=\"sk-top-container\"><div class=\"sk-text-repr-fallback\"><pre>GaussianNB()</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class=\"sk-container\" hidden><div class=\"sk-item\"><div class=\"sk-estimator sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"sk-estimator-id-1\" type=\"checkbox\" checked><label for=\"sk-estimator-id-1\" class=\"sk-toggleable__label sk-toggleable__label-arrow\">GaussianNB</label><div class=\"sk-toggleable__content\"><pre>GaussianNB()</pre></div></div></div></div></div>"
      ],
      "text/plain": [
       "GaussianNB()"
      ]
     },
     "execution_count": 9,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "model.fit(X_train,y_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "c11044d6",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1,\n",
       "       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,\n",
       "       1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])"
      ]
     },
     "execution_count": 10,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "y_pred=model.predict(X_test)\n",
    "y_pred"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "e68ec1df",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "0.9516129032258065\n"
     ]
    }
   ],
   "source": [
    "#accuracy\n",
    "\n",
    "print(model.score(X_test, y_test))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "id": "3395b54b",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "0.9516129032258065\n"
     ]
    }
   ],
   "source": [
    "accuracy_NB=accuracy_score(y_test,y_pred)\n",
    "print(accuracy_NB)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "id": "20c60d40",
   "metadata": {},
   "outputs": [],
   "source": [
    "#Decision tree"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "id": "ecff3d70",
   "metadata": {},
   "outputs": [],
   "source": [
    "X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=0.2,random_state=42)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "id": "370bfcf5",
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.tree import DecisionTreeClassifier"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "id": "a0edb1b1",
   "metadata": {},
   "outputs": [],
   "source": [
    "model1=DecisionTreeClassifier()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "id": "e09de2c2",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<style>#sk-container-id-2 {color: black;}#sk-container-id-2 pre{padding: 0;}#sk-container-id-2 div.sk-toggleable {background-color: white;}#sk-container-id-2 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-2 label.sk-toggleable__label-arrow:before {content: \"▸\";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-2 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-2 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-2 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-2 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-2 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-2 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: \"▾\";}#sk-container-id-2 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-2 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-2 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-2 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-2 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-2 div.sk-parallel-item::after {content: \"\";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-2 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-2 div.sk-serial::before {content: \"\";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-2 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-2 div.sk-item {position: relative;z-index: 1;}#sk-container-id-2 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-2 div.sk-item::before, #sk-container-id-2 div.sk-parallel-item::before {content: \"\";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-2 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-2 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-2 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-2 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-2 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-2 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-2 div.sk-label-container {text-align: center;}#sk-container-id-2 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-2 div.sk-text-repr-fallback {display: none;}</style><div id=\"sk-container-id-2\" class=\"sk-top-container\"><div class=\"sk-text-repr-fallback\"><pre>DecisionTreeClassifier()</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class=\"sk-container\" hidden><div class=\"sk-item\"><div class=\"sk-estimator sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"sk-estimator-id-2\" type=\"checkbox\" checked><label for=\"sk-estimator-id-2\" class=\"sk-toggleable__label sk-toggleable__label-arrow\">DecisionTreeClassifier</label><div class=\"sk-toggleable__content\"><pre>DecisionTreeClassifier()</pre></div></div></div></div></div>"
      ],
      "text/plain": [
       "DecisionTreeClassifier()"
      ]
     },
     "execution_count": 17,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "model1.fit(X_train,y_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "id": "352a79a6",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,\n",
       "       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,\n",
       "       1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])"
      ]
     },
     "execution_count": 18,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "y_pred=model1.predict(X_test)\n",
    "y_pred"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "id": "2aec9a40",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "0.967741935483871\n"
     ]
    }
   ],
   "source": [
    "accuracy_DT=accuracy_score(y_test,y_pred)\n",
    "print(accuracy_DT)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "id": "760e3690",
   "metadata": {},
   "outputs": [],
   "source": [
    "#Random Forest"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "id": "26f2958d",
   "metadata": {},
   "outputs": [],
   "source": [
    "X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=0.3,random_state=42)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "id": "a09eba8b",
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.ensemble import RandomForestClassifier"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "id": "b6afd667",
   "metadata": {},
   "outputs": [],
   "source": [
    "model2=RandomForestClassifier()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "id": "71a5c53a",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<style>#sk-container-id-3 {color: black;}#sk-container-id-3 pre{padding: 0;}#sk-container-id-3 div.sk-toggleable {background-color: white;}#sk-container-id-3 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-3 label.sk-toggleable__label-arrow:before {content: \"▸\";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-3 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-3 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-3 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-3 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-3 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-3 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: \"▾\";}#sk-container-id-3 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-3 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-3 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-3 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-3 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-3 div.sk-parallel-item::after {content: \"\";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-3 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-3 div.sk-serial::before {content: \"\";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-3 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-3 div.sk-item {position: relative;z-index: 1;}#sk-container-id-3 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-3 div.sk-item::before, #sk-container-id-3 div.sk-parallel-item::before {content: \"\";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-3 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-3 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-3 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-3 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-3 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-3 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-3 div.sk-label-container {text-align: center;}#sk-container-id-3 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-3 div.sk-text-repr-fallback {display: none;}</style><div id=\"sk-container-id-3\" class=\"sk-top-container\"><div class=\"sk-text-repr-fallback\"><pre>RandomForestClassifier()</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class=\"sk-container\" hidden><div class=\"sk-item\"><div class=\"sk-estimator sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"sk-estimator-id-3\" type=\"checkbox\" checked><label for=\"sk-estimator-id-3\" class=\"sk-toggleable__label sk-toggleable__label-arrow\">RandomForestClassifier</label><div class=\"sk-toggleable__content\"><pre>RandomForestClassifier()</pre></div></div></div></div></div>"
      ],
      "text/plain": [
       "RandomForestClassifier()"
      ]
     },
     "execution_count": 24,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "model2.fit(X_train,y_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "id": "92cf4236",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1,\n",
       "       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,\n",
       "       1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,\n",
       "       1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1,\n",
       "       1, 1, 1, 1, 1])"
      ]
     },
     "execution_count": 25,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "y_pred=model2.predict(X_test)\n",
    "y_pred"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "id": "5e32b655",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "0.967741935483871\n"
     ]
    }
   ],
   "source": [
    "accuracy_RF=accuracy_score(y_test,y_pred)\n",
    "print(accuracy_DT)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "id": "8d3b9b83",
   "metadata": {},
   "outputs": [],
   "source": [
    "#SVM"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "id": "690467c6",
   "metadata": {},
   "outputs": [],
   "source": [
    "X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=0.2,random_state=42)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "id": "f9eb9e81",
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.svm import SVC"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "id": "d7d2407c",
   "metadata": {},
   "outputs": [],
   "source": [
    "model3=SVC()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "id": "e9713377",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<style>#sk-container-id-4 {color: black;}#sk-container-id-4 pre{padding: 0;}#sk-container-id-4 div.sk-toggleable {background-color: white;}#sk-container-id-4 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-4 label.sk-toggleable__label-arrow:before {content: \"▸\";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-4 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-4 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-4 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-4 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-4 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-4 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: \"▾\";}#sk-container-id-4 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-4 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-4 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-4 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-4 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-4 div.sk-parallel-item::after {content: \"\";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-4 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-4 div.sk-serial::before {content: \"\";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-4 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-4 div.sk-item {position: relative;z-index: 1;}#sk-container-id-4 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-4 div.sk-item::before, #sk-container-id-4 div.sk-parallel-item::before {content: \"\";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-4 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-4 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-4 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-4 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-4 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-4 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-4 div.sk-label-container {text-align: center;}#sk-container-id-4 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-4 div.sk-text-repr-fallback {display: none;}</style><div id=\"sk-container-id-4\" class=\"sk-top-container\"><div class=\"sk-text-repr-fallback\"><pre>SVC()</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class=\"sk-container\" hidden><div class=\"sk-item\"><div class=\"sk-estimator sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"sk-estimator-id-4\" type=\"checkbox\" checked><label for=\"sk-estimator-id-4\" class=\"sk-toggleable__label sk-toggleable__label-arrow\">SVC</label><div class=\"sk-toggleable__content\"><pre>SVC()</pre></div></div></div></div></div>"
      ],
      "text/plain": [
       "SVC()"
      ]
     },
     "execution_count": 31,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "model3.fit(X_train,y_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 32,
   "id": "33b692b5",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,\n",
       "       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,\n",
       "       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])"
      ]
     },
     "execution_count": 32,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "y_pred=model3.predict(X_test)\n",
    "y_pred"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 33,
   "id": "d928d790",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "0.967741935483871\n"
     ]
    }
   ],
   "source": [
    "accuracy_SVM=accuracy_score(y_test,y_pred)\n",
    "print(accuracy_SVM)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 34,
   "id": "185d9113",
   "metadata": {},
   "outputs": [],
   "source": [
    "#KNN"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 35,
   "id": "46cc1d97",
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.neighbors import KNeighborsClassifier"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 36,
   "id": "5d86db19",
   "metadata": {},
   "outputs": [],
   "source": [
    "model4=KNeighborsClassifier(n_neighbors=3)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 37,
   "id": "26923e48",
   "metadata": {},
   "outputs": [],
   "source": [
    "X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=0.2,random_state=42)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 38,
   "id": "080071c6",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<style>#sk-container-id-5 {color: black;}#sk-container-id-5 pre{padding: 0;}#sk-container-id-5 div.sk-toggleable {background-color: white;}#sk-container-id-5 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-5 label.sk-toggleable__label-arrow:before {content: \"▸\";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-5 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-5 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-5 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-5 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-5 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-5 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: \"▾\";}#sk-container-id-5 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-5 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-5 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-5 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-5 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-5 div.sk-parallel-item::after {content: \"\";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-5 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-5 div.sk-serial::before {content: \"\";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-5 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-5 div.sk-item {position: relative;z-index: 1;}#sk-container-id-5 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-5 div.sk-item::before, #sk-container-id-5 div.sk-parallel-item::before {content: \"\";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-5 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-5 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-5 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-5 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-5 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-5 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-5 div.sk-label-container {text-align: center;}#sk-container-id-5 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-5 div.sk-text-repr-fallback {display: none;}</style><div id=\"sk-container-id-5\" class=\"sk-top-container\"><div class=\"sk-text-repr-fallback\"><pre>KNeighborsClassifier(n_neighbors=3)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class=\"sk-container\" hidden><div class=\"sk-item\"><div class=\"sk-estimator sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"sk-estimator-id-5\" type=\"checkbox\" checked><label for=\"sk-estimator-id-5\" class=\"sk-toggleable__label sk-toggleable__label-arrow\">KNeighborsClassifier</label><div class=\"sk-toggleable__content\"><pre>KNeighborsClassifier(n_neighbors=3)</pre></div></div></div></div></div>"
      ],
      "text/plain": [
       "KNeighborsClassifier(n_neighbors=3)"
      ]
     },
     "execution_count": 38,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "model4.fit(X_train,y_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 39,
   "id": "00b5b110",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1,\n",
       "       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,\n",
       "       1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])"
      ]
     },
     "execution_count": 39,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "y_pred=model4.predict(X_test)\n",
    "y_pred"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 40,
   "id": "7633eca9",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "0.9516129032258065\n"
     ]
    }
   ],
   "source": [
    "accuracy_KNN=accuracy_score(y_test,y_pred)\n",
    "print(accuracy_KNN)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 41,
   "id": "1ed10e70",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "[<matplotlib.lines.Line2D at 0x1acb1bf7510>]"
      ]
     },
     "execution_count": 41,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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3o/VPSeIGcXJkye1c+dO7d69Ww0NDVq9erVKS0vV1NQ00UObMm7cuKHFixfra1/72ojvf+lLX1JlZaW+9rWv6aWXXtLs2bP10Y9+1NorC/Y9//zz2rFjh1588UXV1NSop6dHJSUlunHjhnUO9z328vLy9Nhjj+nMmTM6c+aMfuVXfkUbNmyw/o+bex5/L730kg4dOqS77ror5Dj3Pj5+6Zd+Sc3Nzdbr5Zdftt6L2T03ENY999xjbNu2LeTYnXfeaXzuc5+boBFNbZKM73znO9bPvb29xuzZs43HHnvMOtbV1WVkZWUZf/M3fzMBI5yarl69akgynn/+ecMwuO+JNHPmTOPIkSPc8wS4fv268e53v9uoqakxPvShDxkPPfSQYRj89x4vn//8543FixeP+F4s7zmZmDACgYDq6+tVUlIScrykpES1tbUTNKrU8tprr6mlpSXk78Dj8ehDH/oQfwcx1N7eLkmaNWuWJO57IgSDQZ04cUI3btzQihUruOcJsGPHDt177736yEc+EnKcex8/r7zyiubOnav58+dr48aNevXVVyXF9p5PmQ0gY621tVXBYFC5ubkhx3Nzc9XS0jJBo0ot5n0e6e/g9ddfn4ghTTmGYWjXrl365V/+ZS1atEgS9z2eXn75Za1YsUJdXV2aMWOGvvOd7+i9732v9X/c3PP4OHHihP7rv/5LL7300rD3+O89PpYtW6Zjx47pPe95j95880194Qtf0MqVK3X+/PmY3nOCmDE4HI6Qnw3DGHYM8cXfQfx85jOf0U9/+lO98MILw97jvsfewoULdfbsWb399tv69re/rU9+8pN6/vnnrfe557F3+fJlPfTQQ3r22Wfl9XrDnse9j63S0lLrz+973/u0YsUK3XHHHXr66ae1fPlySbG550wnhZGTkyOXyzUs63L16tVh0SPiw6xk5+8gPv7wD/9Q3/3ud/WjH/1IeXl51nHue/ykp6frXe96l5YsWaKKigotXrxYf/3Xf809j6P6+npdvXpVxcXFSktLU1pamp5//nl99atfVVpamnV/uffxNX36dL3vfe/TK6+8EtP/3gliwkhPT1dxcbFqampCjtfU1GjlypUTNKrUMn/+fM2ePTvk7yAQCOj555/n72AcDMPQZz7zGT3zzDP693//d82fPz/kfe574hiGIb/fzz2Po1/91V/Vyy+/rLNnz1qvJUuW6GMf+5jOnj2rBQsWcO8TwO/363/+5380Z86c2P73HkXRcco4ceKE4Xa7jaeeesq4cOGCsXPnTmP69OlGY2PjRA9tyrh+/brR0NBgNDQ0GJKMyspKo6GhwXj99dcNwzCMxx57zMjKyjKeeeYZ4+WXXzZ+7/d+z5gzZ47R0dExwSNPXp/+9KeNrKws47nnnjOam5utV2dnp3UO9z32ysvLjR//+MfGa6+9Zvz0pz81/vzP/9xwOp3Gs88+axgG9zyRBq9OMgzufTx89rOfNZ577jnj1VdfNV588UVj3bp1RkZGhvX8jNU9J4gZw4EDB4yCggIjPT3d+MAHPmAtQ0Vs/OhHPzIkDXt98pOfNAyjbyne5z//eWP27NmGx+MxPvjBDxovv/zyxA46yY10vyUZf/u3f2udw32Pvd///d+3/r/k1ltvNX71V3/VCmAMg3ueSEODGO597JWVlRlz5swx3G63MXfuXOO3fuu3jPPnz1vvx+qeOwzDMGKQKQIAAEgoamIAAEBSIogBAABJiSAGAAAkJYIYAACQlAhiAABAUiKIAQAASYkgBgAAJCWCGAAAkJQIYgAAQFIiiAEAAEmJIAYAACQlghgAAJCU/n8dheuWkz9DIAAAAABJRU5ErkJggg==",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "acc=[]\n",
    "for i in range(1,50):\n",
    "    m=KNeighborsClassifier(n_neighbors=i)\n",
    "    m.fit(X_train,y_train)\n",
    "    y_pred=m.predict(X_test)\n",
    "    acc.append(accuracy_score(y_test,y_pred))\n",
    "plt.plot(acc)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 42,
   "id": "2e98dc4a",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.9516129032258065"
      ]
     },
     "execution_count": 42,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "acc[5]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 43,
   "id": "384ac106",
   "metadata": {},
   "outputs": [],
   "source": [
    "model4=KNeighborsClassifier(n_neighbors=5)\n",
    "X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=0.2,random_state=42)\n",
    "model4.fit(X_train,y_train)\n",
    "y_pred=model4.predict(X_test)\n",
    "accuracy_KNN=accuracy_score(y_test,y_pred)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 44,
   "id": "04afc758",
   "metadata": {},
   "outputs": [],
   "source": [
    "#Logistic Resgression"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 45,
   "id": "d96a215e",
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.linear_model import LogisticRegression"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 46,
   "id": "5c09a933",
   "metadata": {},
   "outputs": [],
   "source": [
    "model5=LogisticRegression()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 47,
   "id": "90082134",
   "metadata": {},
   "outputs": [],
   "source": [
    "X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=0.3,random_state=42)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 48,
   "id": "4058549a",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\anike\\anaconda3\\Lib\\site-packages\\sklearn\\linear_model\\_logistic.py:460: ConvergenceWarning: lbfgs failed to converge (status=1):\n",
      "STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.\n",
      "\n",
      "Increase the number of iterations (max_iter) or scale the data as shown in:\n",
      "    https://scikit-learn.org/stable/modules/preprocessing.html\n",
      "Please also refer to the documentation for alternative solver options:\n",
      "    https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression\n",
      "  n_iter_i = _check_optimize_result(\n"
     ]
    },
    {
     "data": {
      "text/html": [
       "<style>#sk-container-id-6 {color: black;}#sk-container-id-6 pre{padding: 0;}#sk-container-id-6 div.sk-toggleable {background-color: white;}#sk-container-id-6 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-6 label.sk-toggleable__label-arrow:before {content: \"▸\";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-6 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-6 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-6 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-6 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-6 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-6 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: \"▾\";}#sk-container-id-6 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-6 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-6 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-6 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-6 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-6 div.sk-parallel-item::after {content: \"\";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-6 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-6 div.sk-serial::before {content: \"\";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-6 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-6 div.sk-item {position: relative;z-index: 1;}#sk-container-id-6 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-6 div.sk-item::before, #sk-container-id-6 div.sk-parallel-item::before {content: \"\";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-6 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-6 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-6 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-6 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-6 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-6 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-6 div.sk-label-container {text-align: center;}#sk-container-id-6 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-6 div.sk-text-repr-fallback {display: none;}</style><div id=\"sk-container-id-6\" class=\"sk-top-container\"><div class=\"sk-text-repr-fallback\"><pre>LogisticRegression()</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class=\"sk-container\" hidden><div class=\"sk-item\"><div class=\"sk-estimator sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"sk-estimator-id-6\" type=\"checkbox\" checked><label for=\"sk-estimator-id-6\" class=\"sk-toggleable__label sk-toggleable__label-arrow\">LogisticRegression</label><div class=\"sk-toggleable__content\"><pre>LogisticRegression()</pre></div></div></div></div></div>"
      ],
      "text/plain": [
       "LogisticRegression()"
      ]
     },
     "execution_count": 48,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "model5.fit(X_train,y_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 49,
   "id": "75eac863",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,\n",
       "       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,\n",
       "       1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,\n",
       "       1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1,\n",
       "       1, 1, 0, 1, 1])"
      ]
     },
     "execution_count": 49,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "y_pred=model5.predict(X_test)\n",
    "y_pred"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 50,
   "id": "ae757202",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "0.978494623655914\n"
     ]
    }
   ],
   "source": [
    "accuracy_LR=accuracy_score(y_test,y_pred)\n",
    "print(accuracy_LR)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 51,
   "id": "a65f299d",
   "metadata": {},
   "outputs": [],
   "source": [
    "#plot"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 52,
   "id": "0720287c",
   "metadata": {},
   "outputs": [],
   "source": [
    "y_plot=[accuracy_NB,accuracy_DT,accuracy_RF,accuracy_SVM,accuracy_KNN,accuracy_LR]\n",
    "X_plot=['NB','DT','RF','SVM','KNN','LR']\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 53,
   "id": "944191a4",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAkkAAAGwCAYAAAC99fF4AAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjcuMSwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy/bCgiHAAAACXBIWXMAAA9hAAAPYQGoP6dpAABeR0lEQVR4nO3deXhTVd4H8G+Spm260o22dC9lKTsU6EZVnAFEXBhkBEEUKbwojOIwzivoqwLDgOKI2ygjS1lcBhQZlxlUcFTsAlTLvghl6b7RfU/T5L5/pAnUBmhKkpvl+3mePg+9Obn3l0Ob/HruOb8jEQRBABERERF1IhU7ACIiIiJrxCSJiIiIyAAmSUREREQGMEkiIiIiMoBJEhEREZEBTJKIiIiIDGCSRERERGSAk9gB2CqNRoOSkhJ4enpCIpGIHQ4RERF1gyAIaGhoQJ8+fSCV3nisiElSD5WUlCAsLEzsMIiIiKgHCgsLERoaesM2TJJ6yNPTE4C2k728vEx6bpVKhX379mHixImQy+UmPTddxX62DPazZbCfLYP9bDnm6uv6+nqEhYXpP8dvhElSD+lusXl5eZklSXJzc4OXlxd/Cc2I/WwZ7GfLYD9bBvvZcszd192ZKsOJ20REREQGMEkiIiIiMoBJEhEREZEBTJKIiIiIDGCSRERERGQAkyQiIiIiA5gkERERERnAJImIiIjIACZJRERERAYwSSIiIiKrolYDBw5I8OOPIThwQAK1Wpw4mCQRERGR1dizB4iMBCZMcML69aMxYYITIiO1xy2NSRIRERFZhT17gOnTgaKizseLi7XHLZ0oMUkiIiIi0anVwJIlgCAAEhcV5L3rAAgAtMcA4OmnYdFbb0ySiIiISHTp6VdHkNxiytHnsQz0npGtf1wQgMJCbTtLYZJEREREoistvfpv14hKAEBbmfcN25kbkyQiIiISXXCw7l8CXCO1SVJrnv8N2pkfkyQiIiISXUoKEBoKOPs3wslTCY1KitYiH/3jEgkQFqZtZylMkoiIiEh0Mhnw5pvQjyIpi3wBtQyANkECgDfe0LazFCZJREREZBWmTQOSp3W91RYaCuzerX3ckpwsezkiIiIiw9raNchrqQIAvL7MB6czfsbkySMwfryTRUeQdJgkERERkVU4WlCD5jY1/NydMftuD3wtLcbttw8XJUECeLuNiIiIrETmBe2ttqQYf0ilEpGjYZJEREREViK9I0lKiem69F8MTJKIiIhIdHUtKhwvrAUAJPdjkkREREQEADh0qQoaAYj2d0dIL4XY4QBgkkRERERWICNXe6ttnJWMIgFMkoiIiMgKZHTMRxpnJfORACZJREREJLKimmZcrmyCTCpBQl8/scPRY5JEREREotIt/R8e6g0vV7nI0VzFJImIiIhEla6fjxQgciSdMUkiIiIi0Wg0ArIuarciSbGiSdsAkyQiIiIS0ZnSelQ3tcHdWYYRYb3EDqcTJklEREQkGt2qtoRoP8hl1pWWWFc0RERE5FCssT6SDpMkIiIiEkWrSo3svGoA1jcfCWCSRERERCL5Oa8Gbe0aBHq5oG+Ah9jhdMEkiYiIiESRfuEKAGBcTAAkEonI0XTFJImIiIhEoZuPZI232gAmSURERCSC6qY2nC6pBwAkxVjPViTXYpJEREREFqfbimRgkCd6e7qKHI1hTJKIiIjI4vRL/2Os81YbwCSJiIiILEwQBH0RSWusj6TDJImIiIgsKq+qGcW1LXCWSTE2ylfscK6LSRIRERFZVEaudun/qIhecHN2Ejma62OSRERERBaVrl/6HyByJDfGJImIiIgspl2twcFLVQCAZCuetA0wSSIiIiILOlFch4bWdngr5Bga4i12ODfEJImIiIgsRrf0P6mvH2RS69uK5FpMkoiIiMhibGHpvw6TJCIiIrKIJmU7jhbUALDuIpI6TJKIiIjIIg5froJKLSDMV4EIP3exw7kpJklERERkEen6rUise+m/DpMkIiIisgjdpra2cKsNYJJEREREFlBe34rz5Y2QSLQr22yB6EnSu+++i6ioKLi6uiIuLg7p6ek3bP/OO+8gNjYWCoUCAwYMwI4dOzo9fscdd0AikXT5mjJlir7NihUrujweFBRkltdHREREV5f+Dw3xho+7s8jRdI+oG6bs2rULTz/9NN59910kJyfjvffew+TJk3HmzBmEh4d3ab9hwwYsX74cmzZtwpgxY5CdnY0FCxbAx8cH9957LwBgz549aGtr0z+nqqoKw4cPx+9///tO5xo8eDC+/fZb/fcymcxMr5KIiIh0t9qsvcr2tURNktavX4/U1FTMnz8fAPDGG2/gm2++wYYNG7B27dou7d9//30sXLgQM2bMAABER0fj0KFDeOWVV/RJkq9v592Ed+7c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Rv/Vq1cvcYK1Aa6urpgwYQJefPFFZGVlYe7cufqRZ29vb0yfPl1/y23r1q2YPn06vLy8DJ5r9uzZOHToEFasWIFHHnkETk5OFnsd1i4kJAQHDhxAaWkp7rrrrusmSn379sWCBQuwbNmym66wevnll7Fr1y4uTLhFuveLxMREfPrpp3j99dfx7bffWuz6TJKswMsvv4wvv/wSWVlZN20rk8nQ0tJigajs23fffYeTJ08anHtApvPSSy/hb3/7G0pKSsQOxS4MGjQITU1N+u9TU1ORmZmJf//738jMzDR4q03H19cX9913Hw4cOMBbbQaEh4fjwIEDqKiowMSJE1FfX2+w3Ysvvojz589j586dNzzf2LFjMW3atJvenqPu8/HxwZNPPolnnnnGYmUAmCRZgaFDh2L27Nl4++23uzxWUVGBsrIy5Ofn45NPPsH777+P+++/X4QobZdSqURZWRmKi4tx5MgRrFmzBvfffz/uuece3uoxszvuuAODBw/GmjVrxA7FplRVVeHOO+/EBx98gBMnTuDy5cv45JNPsG7duk6//7fffjtiYmLwyCOPICYm5qbzFbdt24bKykoMHDjQ3C/BJoWGhuKHH35AVVUVJk6ciLq6ui5tAgMDsXTp0m7NpfvrX/+K7777DufOnTNHuHajrq6u0y34Y8eOoaCgwGDbxYsX49y5c/j0008tEhuTJCvxl7/8xWBmPGDAAAQHByMmJgbPPvssFi5caDCZouv7+uuvERwcjMjISNx11134/vvv8dZbb+Hzzz+HTCYTOzy7t3TpUmzatAmFhYVih2IzPDw8EB8fj9dffx233XYbhgwZghdeeAELFizA3//+905t582bh5qamm6NDikUCvj5+ZkrbLugu/VWW1uLCRMmoLa2tkubP//5z/Dw8Ljpufr374958+ahtbXVDJHajx9++AEjR47s9PXiiy8abBsQEIA5c+ZgxYoV0Gg0Zo9NIliydCURERGRjeBIEhEREZEBTJKIiIiIDGCSRERERGQAkyQiIiIiA5gkERERERnAJImIiIjIACZJRERERAYwSSIiIiIygEkSEVE3/fDDD5BIJAarMF9PZGQk3njjDbPFRETmwySJiOzG3LlzIZFI8Pjjj3d5bNGiRZBIJJg7d67lAyMim8QkiYjsSlhYGHbu3ImWlhb9sdbWVvzzn/9EeHi4iJERka1hkkREdmXUqFEIDw/Hnj179Mf27NmDsLAwjBw5Un9MqVTiqaeeQu/eveHq6opx48bhp59+6nSuvXv3on///lAoFBg/fjzy8vK6XC8rKwu33XYbFAoFwsLC8NRTT6Gpqclsr4+ILIdJEhHZncceewxbt27Vf5+WloZ58+Z1avO///u/+PTTT7F9+3YcOXIEMTExmDRpEqqrqwEAhYWFmDZtGu6++24cO3YM8+fPx7Jlyzqd4+TJk5g0aRKmTZuGEydOYNeuXcjIyMAf/vAH879IIjI7JklEZHfmzJmDjIwM5OXlIT8/H5mZmXj44Yf1jzc1NWHDhg149dVXMXnyZAwaNAibNm2CQqHAli1bAAAbNmxAdHQ0Xn/9dQwYMACzZ8/uMp/p1VdfxaxZs/D000+jX79+SEpKwltvvYUdO3agtbXVki+ZiMzASewAiIhMzd/fH1OmTMH27dshCAKmTJkCf39//eMXL16ESqVCcnKy/phcLsfYsWNx9uxZAMDZs2eRkJAAiUSib5OYmNjpOjk5Obhw4QI+/PBD/TFBEKDRaHD58mXExsaa6yUSkQUwSSIiuzRv3jz9ba933nmn02OCIABApwRId1x3TNfmRjQaDRYuXIinnnqqy2OcJE5k+3i7jYjs0l133YW2tja0tbVh0qRJnR6LiYmBs7MzMjIy9MdUKhV+/vln/ejPoEGDcOjQoU7P+/X3o0aNwunTpxETE9Ply9nZ2UyvjIgshUkSEdklmUyGs2fP4uzZs5DJZJ0ec3d3xxNPPIE///nP+Prrr3HmzBksWLAAzc3NSE1NBQA8/vjjuHjxIpYuXYpz587ho48+wrZt2zqd59lnn8XBgwexePFiHDt2DLm5ufjiiy/w5JNPWuplEpEZMUkiIrvl5eUFLy8vg4+9/PLLeOCBBzBnzhyMGjUKFy5cwDfffAMfHx8A2ttln376Kb788ksMHz4c//jHP7BmzZpO5xg2bBgOHDiA3NxcpKSkYOTIkXjhhRcQHBxs9tdGROYnEbpz452IiIjIwXAkiYiIiMgAJklEREREBjBJIiIiIjKASRIRERGRAUySiIiIiAxgkkRERERkAJMkIiIiIgOYJBEREREZwCSJiIiIyAAmSUREREQGMEkiIiIiMuD/AZKXLuMoXd90AAAAAElFTkSuQmCC",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.plot(X_plot, y_plot, 'o', color='blue') \n",
    "plt.plot(X_plot, y_plot)  \n",
    "plt.grid(True)\n",
    "plt.xlabel('Model')\n",
    "plt.ylabel('Accuracy')\n",
    "plt.show()"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.4"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
