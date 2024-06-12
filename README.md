# Hand Gesture Recognition

Excecution order:

1.  collectDataSet.py
   - To collect the images for each gesture that we are going to classify.
2. createDataSet.py
   - From the images that we collected, we extracted the hand landmark information and created 2 lists. One with landmark information of hand and the other with corresponding gesture class. And created a pickle file (**data.pickle**) using 2 lists.
3. modelTraining.py
   - We trained a Random Forest Classification model using the landmark data inside a pickled file(**data.pickl
     e**). And saved the best accurate model (**model.p**) using pickle.
4. modelTesting.py
5. app.py
   - run app.py using the command streamlit run app.py
# App screenshots
![Translator](https://github.com/AnjaliiK/Hand-Gesture-Recognition/assets/79195160/8c5afaa5-2e7d-4d51-a231-9c45db3e28d4)

![Screenshot (120)](https://github.com/AnjaliiK/Hand-Gesture-Recognition/assets/79195160/0cb8fe01-2767-4a58-ad59-4ddfa5aed0fc)

![Screenshot 2024-04-13 230227](https://github.com/AnjaliiK/Hand-Gesture-Recognition/assets/79195160/9ea4e138-f62a-4965-a9ea-82c48437b134)

![out](https://github.com/AnjaliiK/Hand-Gesture-Recognition/assets/79195160/1679a083-2fc4-4d8c-9ac3-0c5b201f480c)


![Screenshot 2024-04-13 222719](https://github.com/AnjaliiK/Hand-Gesture-Recognition/assets/79195160/31db2b55-27ed-49db-8678-8c9910c0a2cf)

![out](https://github.com/AnjaliiK/Hand-Gesture-Recognition/assets/79195160/8bfb49dd-b649-4fe1-8d68-5eaa5a552835)




