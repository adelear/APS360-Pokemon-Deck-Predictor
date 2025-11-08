import pokemon_deck_predictor


# visualize a few images
#pokemon_deck_predictor.visalizeData() #Double checked that the data is visualizing correctly

################## TWO STEPS ############################

############################# STEP 1: Classify where the ID is: Left or Right #########################
# Make CNN model to predict where the ID is--Left or Right 

#making sure the correct number of right side and left side is in the folder
model = pokemon_deck_predictor.LargeNet()
pokemon_deck_predictor.train_net(model, batch_size = 128, learning_rate = 0.01, num_epochs = 30) 
#side = pokemon_deck_predictor.predict_side("model_large_bs32_lr0.001_epoch29", "example_pokemon_img.png")
#pokemon_deck_predictor.readID("example_pokemon_img.png", side) 

############################## STEP 2: Read the ID      #######################################
# When the model predicts it, crop the area of the image that contains the ID
# Feed new cropped region into an OCR model to read the id 
# This will be done when we give it images to read AFTER it is trained


