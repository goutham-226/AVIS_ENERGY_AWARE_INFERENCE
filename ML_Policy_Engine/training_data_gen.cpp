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
#include<algorithm>
#include<cctype>

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
	unsigned int base_gpu_temp;
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

	void Down_Clock_SM(unsigned int min_clock, unsigned int max_clock)
	{
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

	void set_gpu_init_temp(unsigned int t)
	{
		base_gpu_temp = t;
	}

	void reset_gpu_temp()
	{
		while(true)
		{
			int temp = get_gpu_temp();
			if(temp <= base_gpu_temp + 4)
			{
				return;
			}
			this_thread::sleep_for(chrono::milliseconds(500));
		}
	}

	unsigned int get_gpu_temp()
	{
		unsigned int temp_c = 0;
		nvmlReturn_t ret = nvmlDeviceGetTemperature(gpu,NVML_TEMPERATURE_GPU,&temp_c);
		if(ret != NVML_SUCCESS )
		{
			throw runtime_error("Failed to read SM clock");
		}

		return temp_c;
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
  
  int llm_decode(llama_batch& batch,int& tkn_count, int max_tokens)
  {
	int n_past = n_tokens;
	int n_predict = max_tokens;
	llama_sampler* sampler = llama_sampler_chain_init(llama_sampler_chain_default_params());
	llama_sampler_chain_add(sampler,llama_sampler_init_temp(temperature));
	llama_sampler_chain_add(sampler,llama_sampler_init_top_p(top_p,1));
	llama_sampler_chain_add(sampler,llama_sampler_init_top_k(top_k));
	llama_sampler_chain_add(sampler,llama_sampler_init_dist(LLAMA_DEFAULT_SEED));
        
	cout << endl << red << "OUTPUT: " << reset;
	llama_token new_token;
	for(int i=0; i<n_predict; i++)
	{
		new_token = llama_sampler_sample(sampler,context,-1);
		llama_sampler_accept(sampler,new_token);

		tkn_count++;
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
			cout << cyan;
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
  
  string get_chat_template(string& mdl_path, string& system_prompt, string& user_prompt)
  {
    string lower = mdl_path;
	transform(lower.begin(), lower.end(), lower.begin(), [](unsigned char c) { return static_cast<char>(tolower(c)); });
    string formatted = "";
    
    if(lower.find("qwen") != string::npos || lower.find("deepseek") != string::npos)
    {
         formatted = "<|im_start|>system\n" + system_prompt + "<|im_end|>\n <|im_start|>user\n" + user_prompt +"<|im_end|>\n<|im_start|>assistant\n";
    }
    
    else if (lower.find("llama") != string::npos)
    {
        formatted = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n" + system_prompt + "<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n" + user_prompt + "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n";
    }

    else if (lower.find("mistral") != string::npos)
    {
        formatted = "<s>[INST] " + system_prompt + "\n\n" + user_prompt + " [/INST]";
    }

    else if (lower.find("phi") != string::npos)
    {
        formatted = "<|system|>\n" + system_prompt + "<|end|>\n<|user|>\n" + user_prompt + "<|end|>\n<|assistant|>\n";
    }

    else if (lower.find("stablelm") != string::npos)
    {
        formatted = "<|system|>\n" + system_prompt + "\n<|user|>\n" + user_prompt + "\n<|assistant|>\n";
    }

    else if (lower.find("falcon") != string::npos)
    {
        formatted = "System: " + system_prompt + "\nUser: " + user_prompt + "\nAssistant:";
    }

    else
    {
        formatted = system_prompt + "\n\nUser: " + user_prompt + "\nAssistant:";
    }

    return formatted;
  }

  int get_prompt_tokens(const llama_vocab* vocab, string& formatted)
  {
	int tokens = -llama_tokenize(vocab,formatted.c_str(),formatted.size(),nullptr,0,true,true);
	vector<llama_token> tkns(tokens);
	int size = llama_tokenize(vocab,formatted.c_str(),formatted.size(),tkns.data(),tkns.size(),true,true);
	return size;
  }

  string get_model_param_size(string& model_name)
  {
	if(model_name.find("deepseek") != string::npos)
	{
		string ret = "7b";
		return ret;
	}

	if(model_name.find("falcon") != string::npos)
	{
		string ret = "7b";
		return ret;
	}

	if(model_name.find("llama3.2") != string::npos)
	{
		string ret = "3b";
		return ret;
	}

	if(model_name.find("mistral") != string::npos)
	{
		string ret = "7b";
		return ret;
	}

	if(model_name.find("phi3.5") != string::npos)
	{
		string ret = "3b";
		return ret;
	}

	if(model_name.find("qwen2.5-3b") != string::npos)
	{
		string ret = "3b";
		return ret;
	}

	if(model_name.find("qwen2.5-coder") != string::npos)
	{
		string ret = "7b";
		return ret;
	}

	if(model_name.find("stablelm-3b") != string::npos)
	{
		string ret = "3b";
		return ret;
	}

	if(model_name.find("qwen2-7b") != string::npos)
	{
		string ret = "7b";
		return ret;
	}

	return " ";


  }
			
  ~Inference()
  {
	
  }

};
//Inference func end
 
int main(int argc, char** argv)
{
	llama_log_set(llama_log_silencer,nullptr);

	string out_file = "avis-data.csv";
    ofstream file(out_file);
    file << "id , model_size , prompt-tokens , max_tokens , generated_tokens , clock_speed_mhz , latency_s , avg_power_w , energy_j , GPU_tempurature_c" << endl;

    ifstream file_1("/home/goutham/Avis_Inference/ML_Policy_Engine/inputs.txt");

    vector<string> sys_prompts;
    vector<string> user_prompts;
    vector<float> temperature;
    vector<float> top_p;
    vector<float> top_k;
    vector<int> max_new_tokens;
    vector<int> context_len;
    vector<int> clock_speed;
    vector<string> model_path;

    string line;

    while(getline(file_1,line))
    {
        if(line.find("system_prompt=") != string::npos)
        {
            int e = line.find("=");
            string sys_prmpt = line.substr(e+1);
            sys_prompts.push_back(sys_prmpt); 
        }

        if(line.find("prompt=") != string::npos  && line.find("system_prompt=") == string::npos)
        {
            int e = line.find("=");
            string prmpt = line.substr(e+1);
            user_prompts.push_back(prmpt);
        }

        if(line.find("temperature=") != string::npos)
        {
            int e = line.find("=");
            string t = line.substr(e+1);
            float f = stof(t);
            temperature.push_back(f);
        }

        if(line.find("top_p=") != string::npos)
        {
            int e = line.find("=");
            string t = line.substr(e+1);
            float f = stof(t);
            top_p.push_back(f);
        }

        if(line.find("top_k=") != string::npos)
        {
            int e = line.find("=");
            string t = line.substr(e+1);
            float f = stof(t);
            top_k.push_back(f);
        }

        if(line.find("max_new_tokens=") != string::npos)
        {
            int e = line.find("=");
            string t = line.substr(e+1);
            int a = stoi(t);
            max_new_tokens.push_back(a);
        }

        if(line.find("context_len=") != string::npos)
        {
            int e = line.find("=");
            string t = line.substr(e+1);
            int a = stoi(t);
            context_len.push_back(a);
        }

        if(line.find("sm_clock_speed=") != string::npos)
        {
            int e = line.find("=");
            string t = line.substr(e+1);
            int a = stoi(t);
            clock_speed.push_back(a);
        }

        if(line.find("model=") != string::npos)
        {
            int e = line.find("=");
            string t = line.substr(e+1);
            model_path.push_back(t);
        }
    }

    llama_backend_init();

    int len = sys_prompts.size();

	int init_temp = 0;

    for(int i=0; i<len; i++)
    {
        llama_model_params model_params = llama_model_default_params();
        model_params.n_gpu_layers = 99;

        llama_model* model = llama_model_load_from_file(model_path[i].c_str(),model_params);

        llama_context_params context_params = llama_context_default_params();
        context_params.n_ctx = context_len[i];

		llama_context* context = llama_init_from_model(model,context_params);

        const llama_vocab* vocab = llama_model_get_vocab(model);

        Inference inf;
        Timer time;
        powermetrics power;

        string formatted = inf.get_chat_template(model_path[i],sys_prompts[i],user_prompts[i]);

        vector<llama_token> tokens(context_len[i]);
        int n_tokens = llama_tokenize(vocab,formatted.c_str(),formatted.size(),tokens.data(),tokens.size(),true,true);

		inf.set_variables(model,context,vocab,tokens,n_tokens,temperature[i],top_p[i],top_k[i]);

		string log_file_name = "Inference.log";

		unsigned int min_clock = clock_speed[i];
		unsigned int max_clock = clock_speed[i]+200;

		int generated_tokens = 0;

		power.start_power_log(log_file_name);
		uint64_t start = time.time_now();	
		llama_batch batch = inf.llm_prefill();
		power.Down_Clock_SM(min_clock,max_clock);
		power.wait_until_sm_clock_at_or_below(min_clock+100);
		int decode_status = inf.llm_decode(batch,generated_tokens,max_new_tokens[i]); //; genrated tokens
		power.Reset_Clock_SM();
		uint64_t end = time.time_now();
		power.stop_power_log();
		unsigned int gpu_temp = power.get_gpu_temp();

		double Inf_latency = time.latency(start,end);
		Inf_latency = Inf_latency/1000.0; // latency col
		double avg_power_consumption = power.get_avg_power_consumption(log_file_name); // avg power col.
		double energy_used = avg_power_consumption * Inf_latency; // energy col.
		unsigned int clck_sm = min_clock; // clock col
		int prompt_tkns = inf.get_prompt_tokens(vocab,formatted);// no. of prompts

		string model_size = inf.get_model_param_size(model_path[i]); // model size

		// clock_speed_mhz , latency_s , avg_power_w , energy_j , GPU_tempurature_c"
		file << to_string(i) << " , " << model_size << " , " << to_string(prompt_tkns) << " , " << max_new_tokens[i] << " , " << generated_tokens << " , " << clck_sm << " , " << Inf_latency << " , " << avg_power_consumption << " , " << energy_used << " , " << gpu_temp << endl;

		if(i==0)
		{
			power.set_gpu_init_temp(gpu_temp);
		}

		else
		{
			power.reset_gpu_temp();
		}
		
		this_thread::sleep_for(chrono::milliseconds(2000));

		llama_batch_free(batch);
		llama_model_free(model);
		llama_free(context);

    }

	llama_backend_free();
	return 0;

}

