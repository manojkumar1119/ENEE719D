

#include <TensorFlowLite.h>


#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "model.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"

void HandleOutput(tflite::ErrorReporter* error_reporter, float x_value,float y_value);


// Globals, used for compatibility with Arduino-style sketches.
namespace {
tflite::ErrorReporter* error_reporter = nullptr;
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;

constexpr int kTensorArenaSize = 130*1024;
uint8_t tensor_arena[kTensorArenaSize];
byte x1;
float x_float; 
TfLiteTensor* model_input = nullptr;
TfLiteTensor* model_output =nullptr;
}  // namespace

void setup() {
  Serial.begin(9600);
  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;
  // Map the model into a usable data structure. This doesn't involve any
  // copying or parsing, it's a very lightweight operation.
  model = tflite::GetModel(g_model);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    TF_LITE_REPORT_ERROR(error_reporter,"Model provided is schema version %d not equal ""to supported version %d.",model->version(), TFLITE_SCHEMA_VERSION);
    return;
  }

  // This pulls in all the operation implementations we need.
  // NOLINTNEXTLINE(runtime-global-variables)
  static tflite::AllOpsResolver resolver;

  // Build an interpreter to run the model with.
  static tflite::MicroInterpreter static_interpreter(model, resolver, tensor_arena, kTensorArenaSize, error_reporter);
  interpreter = &static_interpreter;

  // Allocate memory from the tensor_arena for the model's tensors.
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {TF_LITE_REPORT_ERROR(error_reporter, "AllocateTensors() failed"); return;}

  // Obtain pointers to the model's input and output tensors.
model_input = interpreter->input(0);
model_output = interpreter->output(0);
if ((model_input->dims->size != 2) || (model_input->dims->data[0] != 16000) ||  (model_input->dims->data[1] !=1) || (model_input->type != kTfLiteInt8))
{TF_LITE_REPORT_ERROR(error_reporter,"Bad input tensor parameters in model");return;}  


}

void loop() {

 
 if (Serial.available()){ 
  Serial.read();
  if (model->version() == TFLITE_SCHEMA_VERSION) {
  Serial.println(model->version());
  }

//if ((model_input->dims->size != 2) || (model_input->dims->data[0] != 16000) ||  (model_input->dims->data[1] !=1) || (model_input->type != kTfLiteInt8))
  Serial.println(model_input->dims->size);
  Serial.println(model_input->dims->data[0]);
  Serial.println(model_input->dims->data[1]);  
  Serial.println(model_input->dims->data[2]); 
  Serial.println(model_input->dims->data[3]); 
  
 for (int i=0;i<16000;i++){
  x1 = Serial.read();
x_float=(x1/127.0) - 1.0;
// Quantize the input from floating-point to integer
int8_t x_quantized=  x_float/ model_input->params.scale +model_input->params.zero_point;
model_input->data.int8[i] =x_quantized; //data.int8 is an integer this is where you put your input
Serial.println(x_float);
 }

  // Run inference, and report any error
  TfLiteStatus invoke_status = interpreter->Invoke();
  if (invoke_status != kTfLiteOk) {TF_LITE_REPORT_ERROR(error_reporter, "Invoke failed");return;}

  // Obtain the quantized output from model's output tensor
   float y[2];
   y[0]=model_output->data.int8[0];
   y[1]=model_output->data.int8[1];
  float out1=(y[0] - model_output->params.zero_point) * model_output->params.scale;
  float out2=(y[1] - model_output->params.zero_point) * model_output->params.scale;

   

  Serial.println(out1);
  Serial.println(out2);
 }

}
