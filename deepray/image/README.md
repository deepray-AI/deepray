# Addons - Image

## Components 
https://www.tensorflow.org/deepray/api_docs/python/dp/image


## Contribution Guidelines
#### Standard API
In order to conform with the current API standard, all image ops
must:
 * Be a standard image processing technique 
 * Must be impossible to implement in one of the other API
 standards (Layers, Losses, etc.).

#### Testing Requirements
 * Simple unittests that demonstrate the image op is behaving as
    expected.
 * To run your `tf.functions` in eager mode and graph mode in the tests, 
   you can use the `@pytest.mark.usefixtures("maybe_run_functions_eagerly")` 
   decorator. This will run the tests twice, once normally, and once
   with `tf.config.run_functions_eagerly(True)`.

#### Documentation Requirements
 * Update the [CODEOWNERS file](https://github.com/deepray-AI/deepray/blob/main/.github/CODEOWNERS)
