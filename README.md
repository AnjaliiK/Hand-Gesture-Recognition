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
