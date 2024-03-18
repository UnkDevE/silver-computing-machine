module SilverBullet

if length(ARGS) != 5
	println("Needs five arguments:
			 the models name saved in JDL2 & BSON 
			 the BSON file where the model struct is saved
			 the file of saved state (Flux) of the model struct in JDL2 
		     the datafile path for the model 
			 and the test data in a folder ready to be containerized")
	exit()
end

model, model_state, model_jdl, model_data, test_data = ARGS

using BSON

BSON.@load model_state model

using Flux, JLD2

model_data = JLD2.load(model, model_state)

using MLDatasets
test = FileDataset([loadfn = FileIO.load,] test_data, pattern = "*", depth = 4)

# batch test into features


# predict model
#
predictions = model(test)

end # module SilverBullet
