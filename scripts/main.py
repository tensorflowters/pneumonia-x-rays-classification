from model_loader import ModelLoader
from basic_model import BasicModel


print("\033[91m============================================================\033[0m")
print("\033[91m****************************LOGS****************************\033[0m")
print("\033[91m============================================================\033[0m\n")

# model = BasicModel()
# model.train(10)

model = ModelLoader()
model.load('notebooks/1_train_validation_test_procedure/model_1.h5')
model.evaluate()
model.predict()

print("\n\033[91m============================================================\033[0m")
print("\033[91m****************************LOGS****************************\033[0m")
print("\033[91m============================================================\033[0m")