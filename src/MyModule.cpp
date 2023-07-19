#include "plugin.hpp"
#include "linalg.h"
using std::pow;

struct MyModule : Module {
	enum ParamId {
		PITCH_PARAM,
		PARAMS_LEN
	};
	enum InputId {
		PITCH_INPUT,
		INPUTS_LEN
	};
	enum OutputId {
		SINE_OUTPUT,
		OUTPUTS_LEN
	};
	enum LightId {
		BLINK_LIGHT,
		LIGHTS_LEN
	};
	
	float phase = 0.f;
	float blinkPhase = 0.f;

	typedef float T;
	size_t N = (size_t)38; size_t one = (size_t)1; size_t two = (size_t)2;
	Mat<T> u, u1, u2, temp1, temp2, temp3, Atemp1, Atemp2, Btemp1, Btemp2, Btemp3;
	SymToeplitzMat<T> A, B, C, Dxx;

	T k, k2, gamma, gamma2, gamma4, K, K2, xi1, xi2, sig1, sig2, Ab, sb, vb, ip, rp, h, h2, h2i, theta, q, qlast, qlast2;
	T M_PI2 = M_PI*M_PI;
	size_t ip_int, rp_int;

	float eps; float tol = 0.0001f;
	int iter; int max_iter = 10;


	MyModule() : u(N-1, one), u1(N-1, one), u2(N-1, one), temp1(N-1, one), temp2(N-1, one), temp3(N-1, one), Atemp1(N-1, one), Atemp2(N-1, one), Btemp1(N-1, one), Btemp2(N-1, one), Btemp3(N-1, one), A(N-1, two), B(N-1, two), C(N-1, two), Dxx(N-1, two) {
		config(PARAMS_LEN, INPUTS_LEN, OUTPUTS_LEN, LIGHTS_LEN);
		configParam(PITCH_PARAM, 0.f, 1.f, 0.f, "");
		configInput(PITCH_INPUT, "");
		configOutput(SINE_OUTPUT, "");
		u.fill(0); u1.fill(0); u2.fill(0); qlast = 0;
	}

	void derive_params(const ProcessArgs& args) {
		k = args.sampleTime; k2 = k*k;
		gamma = 2 * (100.f); gamma2 = gamma*gamma; gamma4 = gamma2*gamma2;
		K = pow((0.05f), 0.5) * gamma / M_PI; K2 = K*K;
		xi1 = (-gamma2 + pow(gamma4 + 4*gamma2*K2*M_PI2, 0.5)) / (2*K2);
		xi2 = (-gamma2 + pow(gamma4 + 1600*gamma2*K2*M_PI2, 0.5)) / (2*K2);
		sig1 = 6*std::log(10)/(xi2-xi1)*(xi2/(5.f) - xi1/(3.f));
		sig2 = 6*std::log(10)/(xi2-xi1)*(-1/(5.f) + 1/(3.f));
		Ab = 1000.f * pow(2 * (100.f) * M_E, 0.5);
		sb = (100.f);
		vb = (0.2f);
		ip_int = std::floor((0.2) * N); ip = (0.2)*N - ip_int;
		rp_int = std::floor((0.8) * N); rp = (0.8)*N - rp_int;
		h = (T)1 / N; h2 = h*h; h2i = (T)1 / h2;
		theta = 0.95;
	}

	void mulB() {
		MAT_AT(B.vals, 0, 0) = 2*theta - 2*k2*gamma2*h2i;
		MAT_AT(B.vals, 1, 0) = (1-theta) + k2*gamma2*h2i;
		MAT_AT(Dxx.vals, 0, 0) = 2*K*k*h2i;
		MAT_AT(Dxx.vals, 1, 0) = -K*k*h2i;
		stmat_mul(Btemp1, B, u1);
		stmat_mul(Btemp2, Dxx, u1);
		stmat_mul(Btemp3, Dxx, Btemp2);
		mat_diff(temp1, Btemp1, Btemp3);
	}

	void mulC() {
		MAT_AT(C.vals, 0, 0) = (sig1*k-theta) + 2*sig2*k*h2i;
		MAT_AT(C.vals, 1, 0) = 0.5*(theta-1) - sig2*k*h2i;
		stmat_mul(temp2, C, u2);
	}

	void solveA() {
		MAT_AT(A.vals, 0, 0) = (theta+sig1*k) + 2*sig2*k*h2i;
		MAT_AT(A.vals, 1, 0) = 0.5*(1-theta) - sig2*k*h2i;
		solve(u, A, temp3, Atemp1, Atemp2);
	}

	void process(const ProcessArgs& args) override {

		this->derive_params(args);
		this->mulB();
		this->mulC();
		mat_sum(temp3, temp1, temp2);
		this->solveA();

		eps = 1.f; iter = 0;
		while (eps > tol && iter <= max_iter) {
			qlast2 = qlast*qlast;
			q = qlast - (Ab*qlast*pow(M_E, -sb*qlast2) + 2*qlast/k + (MAT_AT(u2, ip_int, 0) - MAT_AT(u, ip_int, 0))/k2 + 2*vb/k) / (Ab*(1 - 2*sb*qlast2)*pow(M_E, -sb*qlast2) + 2/k);
			eps = std::abs(q - qlast); iter += 1; qlast = q;
		}

		MAT_AT(u, ip_int, 0) = 2*k*(q+vb) + MAT_AT(u2, ip_int, 0);
		//outputs[SINE_OUTPUT].setVoltage(MAT_AT(Dxx.vals, 1, 0));
		outputs[SINE_OUTPUT].setVoltage(20000.f*MAT_AT(u, rp_int, 0));
		u2.fill_mat(u1); u1.fill_mat(u);

		float pitch = params[PITCH_PARAM].getValue();
		pitch += inputs[PITCH_INPUT].getVoltage();
		pitch = clamp(pitch, -4.f, 4.f);

		float freq = dsp::FREQ_C4 * std::pow(2.f, pitch);

		phase += freq * args.sampleTime;
		if (phase >= 0.5f)
			phase -= 1.f;

		float sine = std::sin(2.f * M_PI * phase);

		//outputs[SINE_OUTPUT].setVoltage(5.f * sine);

		blinkPhase += args.sampleTime;
		if (blinkPhase >= 1.f)
			blinkPhase -= 1.f;

		lights[BLINK_LIGHT].setBrightness(blinkPhase < 0.5 ? 1.f : 0.f);

	}
};


struct MyModuleWidget : ModuleWidget {
	MyModuleWidget(MyModule* module) {
		setModule(module);
		setPanel(createPanel(asset::plugin(pluginInstance, "res/MyModule.svg")));

		addChild(createWidget<ScrewSilver>(Vec(RACK_GRID_WIDTH, 0)));
		addChild(createWidget<ScrewSilver>(Vec(box.size.x - 2 * RACK_GRID_WIDTH, 0)));
		addChild(createWidget<ScrewSilver>(Vec(RACK_GRID_WIDTH, RACK_GRID_HEIGHT - RACK_GRID_WIDTH)));
		addChild(createWidget<ScrewSilver>(Vec(box.size.x - 2 * RACK_GRID_WIDTH, RACK_GRID_HEIGHT - RACK_GRID_WIDTH)));

		addParam(createParamCentered<RoundBlackKnob>(mm2px(Vec(15.24, 46.063)), module, MyModule::PITCH_PARAM));

		addInput(createInputCentered<PJ301MPort>(mm2px(Vec(15.24, 77.478)), module, MyModule::PITCH_INPUT));

		addOutput(createOutputCentered<PJ301MPort>(mm2px(Vec(15.24, 108.713)), module, MyModule::SINE_OUTPUT));

		addChild(createLightCentered<MediumLight<RedLight>>(mm2px(Vec(15.24, 25.81)), module, MyModule::BLINK_LIGHT));
	}
};


Model* modelMyModule = createModel<MyModule, MyModuleWidget>("MyModule");
