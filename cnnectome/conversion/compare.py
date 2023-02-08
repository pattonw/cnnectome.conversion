import torch
import tensorflow as tf
import numpy as np

def compare(tf_path, torch_model, test_data=None):
    with torch.no_grad():
        # no grad to save memory
        torch_input = torch.from_numpy(test_data)
        torch_input = torch_input.unsqueeze(0).unsqueeze(0).float()

        # printing intermediate layer info for debugging is easy, just
        # put print statements in the forward pass of the unet.

        torch_output = torch_model.forward(torch_input)

    with tf.compat.v1.Session() as sess:
        # not sure how to do "no_grad" in tensorflow
        graph = tf.compat.v1.get_default_graph()
        saver = tf.compat.v1.train.import_meta_graph(f"{tf_path}.meta")
        saver.restore(sess, f"{tf_path}")
        tf_input = graph.get_tensor_by_name("Placeholder:0")
        final_up_conv = len(torch_model) == 4
        if final_up_conv:
            tf_output = graph.get_tensor_by_name("Reshape_3:0")
        else:
            tf_output = graph.get_tensor_by_name("Reshape_1:0")
        tf_output_dict = {"out": tf_output}

        """
        # For debugging purposes if you want to print out the intermediate layer outputs
        # Good for figuring out where an operation might have been missed
        for i in range(len(num_down_kernels)):
            name = ""
            for j in range(i + 1):
                name += f"lev{j}/"
            tf_output_dict[f"l{i}"] = name + f"unet_layer_{i}_left_1/Relu:0"
        for i in range(len(num_up_kernels)):
            name = ""
            for j in range(i + 1):
                name += f"lev{j}/"
            tf_output_dict[f"r{i}"] = name + f"unet_layer_{i}_right_1/Relu:0"
            tf_output_dict[f"c{i}"] = name + f"concat:0"
        """

        tf_returned = sess.run(tf_output_dict, feed_dict={tf_input: test_data})

        """
        for i in range(len(num_down_kernels)):
            print(
                f"TF {i} left conv pass range: {(np.min(tf_returned[f'l{i}']), np.max(tf_returned[f'l{i}']))},"
                f" shape: {tf_returned[f'l{i}'].shape}"
            )
        for i in range(len(num_up_kernels)-1, -1, -1):
            print(
                f"TF {i} right input range: {(np.min(tf_returned[f'c{i}']), np.max(tf_returned[f'c{i}']))},"
                f" shape: {tf_returned[f'c{i}'].shape}"
            )
            print(
                f"TF {i} right conv pass range: {(np.min(tf_returned[f'r{i}']), np.max(tf_returned[f'r{i}']))},"
                f" shape: {tf_returned[f'r{i}'].shape}"
            )
        """

    ############################### COMPARE ###################################

    torch_numpy = torch_output.squeeze().detach().numpy()
    tf_numpy = tf_returned["out"]

    max_error = np.max(torch_numpy - tf_numpy)
    assert (
        max_error < 5e-6
    ), f"torch and tensorflow output have a max error of: {max_error}"