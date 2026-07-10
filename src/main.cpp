#include<iostream>
#include<string>
#include<vector>
#include<stdexcept>
#include<chrono>
#include<cstdint>
#include"llama.h"
#include<nvml.h>
#include<fstream>
#include<atomic>
#include<thread>
#include<iomanip>

using namespace std;

#define red "\033[31m"
#define green "\033[32m"
#define yellow "\033[33m"
#define blue "\033[34m"
#define magenta "\033[35m"
#define cyan "\033[36m"
#define reset "\033[0m"


//function to take in logs and only print the specific logs especially one with errors
static void llama_log_silencer(enum ggml_log_level level, const char * text, void * user_data) {
    if (level >= GGML_LOG_LEVEL_ERROR) {
        fprintf(stderr, "%s", text);
    }
}

class powermetrics {
  private:
    nvmlDevice_t gpu;
    atomic<bool> running;
    thread worker;
  public:
    	powermetrics()
    	{
		running.store(false,memory_order_relaxed);
		nvmlReturn_t result = nvmlInit();
		if(result != NVML_SUCCESS)
		{
			throw runtime_error("NVML INIT FAILED!");
		}
		result = nvmlDeviceGetHandleByIndex(0,&gpu);
		if(result != NVML_SUCCESS)
		{
			throw runtime_error("NVML HANDLE_BY_INDEX FAILED!");
		}
   	}

	void start_power_log(const string& filename)
	{
		running.store(true,memory_order_relaxed);
		worker = thread([this,filename](){
		ofstream file(filename);
		file << "power_w , util_gpu_percent , sm_clock_mhz , mem_clock_mhz , memory_used_mb\n";
		while(running)
		{
			unsigned int power_mW = 0;
			unsigned int sm_clock = 0;
			unsigned int mem_clock = 0;

			nvmlUtilization_t util;
			nvmlMemory_t memory;

			nvmlDeviceGetPowerUsage(gpu, &power_mW);
			nvmlDeviceGetUtilizationRates(gpu, &util);
			nvmlDeviceGetClockInfo(gpu, NVML_CLOCK_SM, &sm_clock);
			nvmlDeviceGetClockInfo(gpu, NVML_CLOCK_MEM, &mem_clock);
			nvmlDeviceGetMemoryInfo(gpu, &memory);

			file << "power=" << power_mW/1000.0 << " , " << util.gpu << " , " << sm_clock << " , " << mem_clock << " , "<<memory.used/(1024*1024) << "\n";
			file.flush();

			this_thread::sleep_for(chrono::milliseconds(1));
		}
			
		});
	}

	void stop_power_log()
	{
		running.store(false,memory_order_relaxed);
		if(worker.joinable())
		{
			worker.join();
		}
	}

	void Down_Clock_SM()
	{
    		unsigned int min_clock = 1800;
    		unsigned int max_clock = 2000;
    		nvmlReturn_t result = nvmlDeviceSetGpuLockedClocks(gpu, min_clock, max_clock);
    		if (result != NVML_SUCCESS)
    		{
        		string err = nvmlErrorString(result);
        		throw runtime_error("Failed to lock GPU clocks: " + err);
    		}
	}

	void Reset_Clock_SM()
	{
		unsigned int min_clock = 2600;
		unsigned int max_clock = 2880;
		nvmlReturn_t result = nvmlDeviceSetGpuLockedClocks(gpu, min_clock,max_clock);
		if (result != NVML_SUCCESS)
    		{
        		string err = nvmlErrorString(result);
        		throw runtime_error("Failed to lock GPU clocks: " + err);
    		}


	}

	unsigned int get_sm_clock()
	{
    		unsigned int sm_clock = 0;
    		nvmlReturn_t result = nvmlDeviceGetClockInfo(gpu,NVML_CLOCK_SM,&sm_clock);
	    	if (result != NVML_SUCCESS)
    		{
		   throw runtime_error("Failed to read SM clock");
    		}

    		return sm_clock;
	}

	void wait_until_sm_clock_at_or_below(unsigned int target_max_mhz)
	{

    			const int sleep_ms = 25;
			const int max_ms = 3000;
			int elapsed = 0;
   			const unsigned int tolerance_mhz = 100;
   			while (elapsed <= max_ms)
    			{
        			unsigned int current_clock = get_sm_clock();
        			if (current_clock <= target_max_mhz + tolerance_mhz)
        			{
            				return;
        			}	
        			this_thread::sleep_for(chrono::milliseconds(sleep_ms));
				elapsed += sleep_ms;
    			}
			return;

	}

	void wait_until_sm_clock_at_or_above(unsigned int target_min_mhz)
	{
		const int sleep_ms = 25;
		const int max_ms = 3000;
		int elapsed = 0;
		const unsigned int tolerance_mhz = 100;
		while(elapsed <= max_ms)
		{
			unsigned int current_clock = get_sm_clock();
			if(current_clock >= target_min_mhz - tolerance_mhz)
			{
				return;
			}
			this_thread::sleep_for(chrono::milliseconds(sleep_ms));
			elapsed += sleep_ms;
		}
		return;
	}

	double get_avg_power_consumption(string& filename)
	{
		vector<double> arr;
		ifstream file(filename);
		string line;
		while(getline(file,line))
		{
			if(line.empty())
			{
				continue;
			}

			size_t pw = line.find("power=");
			if(pw == string::npos)
			{
				continue;
			}
			pw += 6;
			size_t end = line.find(",",pw);
			if (end == string::npos)
        		{
				continue;
        		}
			string power_str = line.substr(pw, end - pw);
			double power = stod(power_str);
			arr.push_back(power);

		}
		int n = arr.size();
		int count = 0;
		double sum = 0.0;
		for(int i=0; i<n; i++)
		{
			if(arr[i]!= 0.0)
			{
				sum = sum + arr[i];
				count++;
			}
			
		}

		double avg = sum/(double)count;

		return avg;
	}

	~powermetrics()
	{
		stop_power_log();
		nvmlShutdown();
	}    
};
	
//methods to measure latency
class Timer {
public:
  Timer() = default;
  uint64_t time_now()
  {
  	auto now = chrono::steady_clock::now();
	auto t = chrono::duration_cast<chrono::microseconds>(now.time_since_epoch()).count();
	return static_cast<uint64_t>(t);
  }
  double latency(uint64_t s, uint64_t e)
  {
	double result = ((double)(e - s))/1000.0;
	return result;
  }
		
};
//end

//Inference func start
class Inference {

private:
  llama_model* model;
  llama_context* context;
  const llama_vocab* vocab;
  vector<llama_token> tokens;
  int n_tokens;
  float temperature;
  float top_p;
  float top_k;


public:
  

  void set_variables(llama_model* mdl, llama_context* ctx, const llama_vocab* vcb, vector<llama_token>& tkns, int n_tkns,float temp,float t_p,float t_k)
  {
	model = mdl;
	context = ctx;
	vocab = vcb;
	tokens = tkns;
	n_tokens = n_tkns;
	temperature = temp;
	top_p = t_p;
	top_k = t_k;


  }

  llama_batch llm_prefill()
  {
	llama_batch batch = llama_batch_init(n_tokens,0,1);
	if(batch.token == nullptr)
	{
		throw runtime_error("Could not create batch");
	}
	batch.n_tokens = n_tokens;
	for(int i=0; i<n_tokens; i++)
	{
		batch.token[i] = tokens[i];
		batch.pos[i] = i;
		batch.n_seq_id[i] = 1;
		batch.seq_id[i][0] = 0;
		batch.logits[i] = false;
	}
	batch.logits[n_tokens-1] = true;
	int decode_status = llama_decode(context,batch);
	if(decode_status != 0)
	{
		throw runtime_error("Error couldnt perform llama_decode");
	}
	
	float* logits = llama_get_logits_ith(context,n_tokens-1);
	if(logits == nullptr)
	{
		throw runtime_error("Error couldnt get logits");
	}	

	return batch; 
  
  }
  
  int llm_decode(llama_batch& batch)
  {
	int n_past = n_tokens;
	int n_predict = 10000;
	llama_sampler* sampler = llama_sampler_chain_init(llama_sampler_chain_default_params());
	llama_sampler_chain_add(sampler,llama_sampler_init_temp(temperature));
	llama_sampler_chain_add(sampler,llama_sampler_init_top_p(top_p,1));
	llama_sampler_chain_add(sampler,llama_sampler_init_top_k(top_k));
	llama_sampler_chain_add(sampler,llama_sampler_init_dist(LLAMA_DEFAULT_SEED));
        
	cout << endl << magenta << "OUTPUT: " << reset;
	llama_token new_token;
	for(int i=0; i<n_predict; i++)
	{
		new_token = llama_sampler_sample(sampler,context,-1);
		llama_sampler_accept(sampler,new_token);
		if(llama_vocab_is_eog(vocab,new_token))
		{
			break;
		}
		vector<char> buffer(257);
		int buffer_len = llama_token_to_piece(vocab,new_token,buffer.data(),256,0,false);
		if(buffer_len <= 0)
		{
			cout << "Failed to load buffer" << endl;
			return 1;
		}
		if(buffer_len > 0)
		{
			buffer[buffer_len] = '\0';
			cout << yellow;
			cout.write(buffer.data(), buffer_len);
			cout << flush;
			cout << reset;
		}
		batch.n_tokens = 1;
		batch.token[0] = new_token;
		batch.pos[0] = n_past;
		batch.n_seq_id[0] = 1;
		batch.seq_id[0][0] =0;
		batch.logits[0] = true;
		n_past++;
		int decode_status = llama_decode(context,batch);
		if(decode_status!=0)
		{
			cout << "llama_decode failed during decode phase" << endl;
			return 1;
		}
		float* logits = llama_get_logits_ith(context,0);
		if(logits == nullptr)
		{
			cout << "Failed to get logits during decode " << endl;
			return 1;
		}
		
	}
	llama_sampler_free(sampler);
	return 0;
  }		
			
  ~Inference()
  {
	
  }

};
//Inference func end
 
int main(int argc, char** argv)
{
	cout<< red << "			Welcome to Avis Inference	" << reset << endl;
	cout << red <<"-------------------------------------------------"<<reset<<endl;
	cout << cyan <<"->Enter system prompt:" << reset << endl;
	string system_prompt = "";
	getline(cin,system_prompt);
	cout << cyan <<"->Enter user prompt:" << reset<<endl;
	string prompt = "";
	getline(cin,prompt);
	cout << cyan <<"->Enter tempurature:" << reset << endl;
	float temperature = 0.0;
	cin >> temperature;
	cout << cyan <<"->Enter top_p:" << reset << endl;
	float top_p = 0.0;
	cin >> top_p;
	cout << cyan<<"->Enter top_k:" << reset << endl;
	float top_k = 0.0;
	cin >> top_k;
    cout << red <<"-------------------------------------------------"<<reset<<endl;

	llama_log_set(llama_log_silencer,nullptr);
	llama_backend_init();
	llama_model_params model_params = llama_model_default_params();
	model_params.n_gpu_layers = 99;
	llama_model* model = llama_model_load_from_file("/home/goutham/Avis_Inference/models/llama3.2-3b-instruct-q4_k_m/Llama-3.2-3B-Instruct-Q4_K_M.gguf",model_params);
		
	llama_context_params context_params = llama_context_default_params();
	context_params.n_ctx = 16000;
	
	llama_context* context = llama_init_from_model(model,context_params);
	
	const llama_vocab* vocab = llama_model_get_vocab(model);
		
	string formatted = "<|im_start|>system\n"+system_prompt+"\n<|im_end|>\n<|im_start|>user\n" + prompt + "\n<|im_end|>\n<|im_start|>assistant\n";

	vector<llama_token> tokens(16000);
	int n_tokens = llama_tokenize(vocab,formatted.c_str(),formatted.size(),tokens.data(),16000,true,false);
        
	tokens.resize(n_tokens);

	Inference obj;
	powermetrics power;
	Timer time;

	obj.set_variables(model,context,vocab,tokens,n_tokens,temperature,top_p,top_k);
	string log_file_name = "Avis_Inference_powermetrics.log";

	cout << red << "################# RUNNING A WARMUP RUN #####################" << reset << endl;
	power.start_power_log(log_file_name);
	uint64_t start = time.time_now();	
	llama_batch batch = obj.llm_prefill();

	
	int decode_status = obj.llm_decode(batch);

	uint64_t end = time.time_now();
	power.stop_power_log();

	llama_batch_free(batch);
	llama_free(context);

	context = llama_init_from_model(model,context_params);

	cout << red << "################# RUNNING BASELINE RUN #####################" << reset << endl;

	power.start_power_log(log_file_name);
	start = time.time_now();	
	batch = obj.llm_prefill();

	
	decode_status = obj.llm_decode(batch);

	end = time.time_now();
	power.stop_power_log();

	double Inf_latency = time.latency(start,end);
	Inf_latency = Inf_latency/1000.0;
	double avg_power_consumption = power.get_avg_power_consumption(log_file_name);
	
	double energy_used = avg_power_consumption * Inf_latency;
	
	cout << endl;
	cout << green << " 		BASELINE_REPORT "<<reset<<endl;
	cout <<green << "--------------------------------------------"<<reset<< endl;
	cout<<cyan<<" ->Inference latency = "<<red<<Inf_latency<<" s."<<reset<<endl;
	cout<<cyan<<" ->Infernce Energy consumed = "<<red<<energy_used<<" j."<<reset<<endl;
	cout <<green<<"-----------------------------------------------"<<reset<<endl;

	llama_batch_free(batch);
	llama_free(context);
	context = llama_init_from_model(model,context_params);
	cout << red << "################# RUNNING BASELINE RUN #####################" << reset << endl;
	power.start_power_log(log_file_name);
	start = time.time_now();	
	batch = obj.llm_prefill();
	power.Down_Clock_SM();
	power.wait_until_sm_clock_at_or_below(2000);
	decode_status = obj.llm_decode(batch);
	end = time.time_now();
	power.stop_power_log();
	power.Reset_Clock_SM();
	Inf_latency = time.latency(start,end);
	Inf_latency = Inf_latency/1000.0;
	avg_power_consumption = power.get_avg_power_consumption(log_file_name);
	energy_used = avg_power_consumption * Inf_latency;
	

cout << endl;
cout << green << " 		AVIS_REPORT "<<reset<<endl;
cout <<green << "--------------------------------------------"<<reset<< endl;
cout<<cyan<<" ->Inference latency = "<<red<<Inf_latency<<" s."<<reset<<endl;
cout<<cyan<<" ->Infernce Energy consumed = "<<red<<energy_used<<" j."<<reset<<endl;
cout <<green<<"-----------------------------------------------"<<reset<<endl;

	llama_batch_free(batch);
	llama_free(context);
	llama_model_free(model);
	llama_backend_free();
	return 0;
	
}

