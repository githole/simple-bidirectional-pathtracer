#include <cmath>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <string>
#include <vector>

const double PI = 3.14159265358979323846;
const double INF = 1e20;
const double EPS = 1e-5;
double MaxDepth = 5;

// *** その他の関数 ***
inline double rand01() { return (double)rand()/RAND_MAX; }

// *** データ構造 ***
struct Vec {
	double x, y, z;
	Vec(const double x_ = 0, const double y_ = 0, const double z_ = 0) : x(x_), y(y_), z(z_) {}
	inline Vec operator+(const Vec &b) const {return Vec(x + b.x, y + b.y, z + b.z);}
	inline Vec operator-(const Vec &b) const {return Vec(x - b.x, y - b.y, z - b.z);}
	inline Vec operator*(const double b) const {return Vec(x * b, y * b, z * b);}
	inline Vec operator/(const double b) const {return Vec(x / b, y / b, z / b);}
	inline const double LengthSquared() const { return x*x + y*y + z*z; }
	inline const double Length() const { return sqrt(LengthSquared()); }
};
inline Vec operator*(double f, const Vec &v) { return v * f; }
inline Vec Normalize(const Vec &v) { return v / v.Length(); }
// 要素ごとの積をとる
inline const Vec Multiply(const Vec &v1, const Vec &v2) {
	return Vec(v1.x * v2.x, v1.y * v2.y, v1.z * v2.z);
}
inline const double Dot(const Vec &v1, const Vec &v2) {
	return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
}
inline const Vec Cross(const Vec &v1, const Vec &v2) {
	return Vec((v1.y * v2.z) - (v1.z * v2.y), (v1.z * v2.x) - (v1.x * v2.z), (v1.x * v2.y) - (v1.y * v2.x));
}
typedef Vec Color;
const Color BackgroundColor(0.0, 0.0, 0.0);

struct Ray {
	Vec org, dir;
	Ray(const Vec org_, const Vec &dir_) : org(org_), dir(dir_) {}
};

enum ReflectionType {
	DIFFUSE,    // 完全拡散面。いわゆるLambertian面。
	SPECULAR,   // 理想的な鏡面。
	REFRACTION, // 理想的なガラス的物質。
};

struct Sphere {
	double radius;
	Vec position;
	Color emission, color;
	ReflectionType ref_type;

	Sphere(const double radius_, const Vec &position_, const Color &emission_, const Color &color_, const ReflectionType ref_type_) :
	  radius(radius_), position(position_), emission(emission_), color(color_), ref_type(ref_type_) {}
	// 入力のrayに対する交差点までの距離を返す。交差しなかったら0を返す。
	inline const double intersect(const Ray &ray) {
		Vec o_p = position - ray.org;
		const double b = Dot(o_p, ray.dir), det = b * b - Dot(o_p, o_p) + radius * radius;
		if (det >= 0.0) {
			const double sqrt_det = sqrt(det);
			const double t1 = b - sqrt_det, t2 = b + sqrt_det;
			if (t1 > EPS)		return t1;
			else if(t2 > EPS)	return t2;
		}
		return 0.0;
	}
};

// Lightサブパス、あるいはEyeサブパスを記録するための頂点データ構造
struct Vertex {
	Vec position; // 頂点の位置
	Vec normal; // 頂点位置での法線
	Color brdf; // BRDF(ひとつ前の頂点 -> この頂点 -> 次の頂点)の値
	double probability_to_next_vertex; // この頂点から次の頂点へのパスがどれくらいの確率密度で生成されたかの値。ただし投影立体角あたりの確率。
	int id; // 頂点がどのオブジェクト上にあるか。
	Vertex(const Vec& position_, const double probability_to_next_vertex_, const int id_, const Color &brdf_, const Vec &normal_) :
	position(position_), probability_to_next_vertex(probability_to_next_vertex_), id(id_), brdf(brdf_), normal(normal_) {}
};

// *** レンダリングするシーンデータ ****
// from small ppt
Sphere spheres[] = {
	
	Sphere(1e5, Vec( 1e5+1,40.8,81.6), Color(), Color(0.75, 0.25, 0.25),DIFFUSE),// 左
	Sphere(1e5, Vec(-1e5+99,40.8,81.6),Color(), Color(0.25, 0.25, 0.75),DIFFUSE),// 右
	Sphere(1e5, Vec(50,40.8, 1e5),     Color(), Color(0.75, 0.75, 0.75),DIFFUSE),// 奥
	Sphere(1e5, Vec(50,40.8,-1e5+170), Color(), Color(), DIFFUSE),// 手前
	Sphere(1e5, Vec(50, 1e5, 81.6),    Color(), Color(0.75, 0.75, 0.75),DIFFUSE),// 床
	Sphere(1e5, Vec(50,-1e5+81.6,81.6),Color(), Color(0.75, 0.75, 0.75),DIFFUSE),// 天井
	Sphere(16.5,Vec(27,16.5,47),       Color(), Color(1,1,1)*.99, SPECULAR),// 鏡
	Sphere(16.5,Vec(73,16.5,78),       Color(), Color(1,1,1)*.99, REFRACTION),//ガラス

	Sphere(5.0, Vec(50.0, 75.0, 81.6),Color(12,12,12), Color(), DIFFUSE),//照明
};
const int LightID = 8; // とてもダサイ。一般の場合に拡張するのはそう難しくないけど今回はしない

// *** レンダリング用関数 ***
// シーンとの交差判定関数
inline bool intersect_scene(const Ray &ray, double *t, int *id) {
	const double n = sizeof(spheres) / sizeof(Sphere);
	*t  = INF;
	*id = -1;
	for (int i = 0; i < int(n); i ++) {
		double d = spheres[i].intersect(ray);
		if (d > 0.0 && d < *t) {
			*t  = d;
			*id = i;
		}
	}
	return *t < INF;
}

// BRDFの値を取得する。
// LightサブパスとEyeサブパスを接続する時などに必要
inline double BRDF(int id, Vec pos, Vec in, Vec out) {
	switch (spheres[id].ref_type) {
	case DIFFUSE:
		return (1.0 / PI);
	case SPECULAR: {
	} break;
	case REFRACTION: {
	} break;
	}
	return 0.0; 
}

// シーンを追跡していき、ぶつかった頂点をverticesに記録していく。
// 光源側と視点側からの二方向から追跡し、それぞれで得られた頂点群を適当につなぐことで光源から視点までのパスを生成する。
void trace_scene(const Ray &ray, const int depth, std::vector<Vertex> *vertices) {
	Color col;
	double t; // レイからシーンの交差位置までの距離
	int id;   // 交差したシーン内オブジェクトのID
	if (!intersect_scene(ray, &t, &id))
		return;
	
	const Sphere &obj = spheres[id];
	Vec hitpoint = ray.org + t * ray.dir; // 交差位置
	const Vec normal  = Normalize(hitpoint - obj.position); // 交差位置の法線
	const Vec orienting_normal = Dot(normal, ray.dir) < 0.0 ? normal : (-1.0 * normal); // 交差位置の法線（物体からのレイの入出を考慮）
	
	double rossian_roulette_probability = std::max(obj.color.x, std::max(obj.color.y, obj.color.z));
	if (depth > MaxDepth) {
		if (rand01() >= rossian_roulette_probability) {
			vertices->push_back(Vertex(hitpoint, 0.0, id, 0.0, orienting_normal));
			return;
		}
	} else
		rossian_roulette_probability = 1.0;

	switch (obj.ref_type) {
	case DIFFUSE: {
		// orienting_normalの方向を基準とした正規直交基底(w, u, v)を作る。この基底に対する半球内で次のレイを飛ばす。
		Vec w, u, v;
		w = orienting_normal;
		if (fabs(w.x) > 0.1)
			u = Normalize(Cross(Vec(0.0, 1.0, 0.0), w));
		else
			u = Normalize(Cross(Vec(1.0, 0.0, 0.0), w));
		v = Cross(w, u);
		// コサイン項を使った重点的サンプリング
		const double r1 = 2 * PI * rand01();
		const double r2 = rand01(), r2s = sqrt(r2);
		Vec dir = Normalize((u * cos(r1) * r2s + v * sin(r1) * r2s + w * sqrt(1.0 - r2)));
		
		// 次の頂点へのパスの生成確率は　ロシアンルーレットの確率 * pdf⊥(ω)になる。
		// pdf⊥(ω) = pdf(ω) / cosθで、pdf(ω) = cosθ/π（コサイン項による重点サンプリング）なので、
		// rossian_roulette_probability / πとなる
		vertices->push_back(Vertex(hitpoint, rossian_roulette_probability * (1.0 / PI), id, obj.color * (1.0 / PI), orienting_normal));
		trace_scene(Ray(hitpoint, dir), depth+1, vertices);
		return;
	} break;
	case SPECULAR: {
		// 完全鏡面なのでレイの反射方向は決定的。
		const Ray reflection_ray = Ray(hitpoint, ray.dir - normal * 2.0 * Dot(normal, ray.dir));
		vertices->push_back(Vertex(hitpoint, rossian_roulette_probability, id, obj.color, orienting_normal));
		trace_scene(reflection_ray, depth+1, vertices);
		return;
	} break;
	case REFRACTION: {
		const Ray reflection_ray = Ray(hitpoint, ray.dir - normal * 2.0 * Dot(normal, ray.dir));
		const double cost = Dot(orienting_normal, reflection_ray.dir);
		bool into = Dot(normal, orienting_normal) > 0.0; // レイがオブジェクトから出るのか、入るのか

		// Snellの法則
		const double nc = 1.0; // 真空の屈折率
		const double nt = 1.5; // オブジェクトの屈折率
		const double nnt = into ? nc / nt : nt / nc;
		const double ddn = Dot(ray.dir, orienting_normal);
		const double cos2t = 1.0 - nnt * nnt * (1.0 - ddn * ddn);
		
		if (cos2t < 0.0) { // 全反射した
			vertices->push_back(Vertex(hitpoint, rossian_roulette_probability, id, obj.color, orienting_normal));
			trace_scene(reflection_ray, depth+1, vertices);
			return ;
		}
		// 屈折していく方向
		Vec tdir = Normalize(ray.dir * nnt - normal * (into ? 1.0 : -1.0) * (ddn * nnt + sqrt(cos2t)));

		// SchlickによるFresnelの反射係数の近似
		const double a = nt - nc, b = nt + nc;
		const double R0 = (a * a) / (b * b);
		const double c = 1.0 - (into ? -ddn : Dot(tdir, normal));
		const double Re = R0 + (1.0 - R0) * pow(c, 5.0);
		const double Tr = 1.0 - Re; // 屈折光の運ぶ光の量
		const double probability  = 0.25 + 0.5 * Re;

		const double cost_ = Dot(-1.0 * orienting_normal, tdir);

		// 屈折と反射のどちらか一方を追跡する。（さもないと指数的にレイが増える）
		// ロシアンルーレットで決定する。
		if (rand01() < probability) { // 反射
			vertices->push_back(Vertex(hitpoint, rossian_roulette_probability * probability, id, obj.color * Re, orienting_normal));
			trace_scene(reflection_ray, depth+1, vertices);
			return;
		} else { // 屈折
			vertices->push_back(Vertex(hitpoint, rossian_roulette_probability * (1.0 - probability), id, obj.color * Tr, -1.0 * orienting_normal));
			trace_scene(Ray(hitpoint, tdir), depth+1, vertices);
			return ;
		}
	} break;
	}
}

// ある頂点v0から別の頂点v1にパスが生成されるとしたとき、その確率
// DIFFUSE面の場合、v0がDiffus面上にあればpdf⊥(ω)なので、上に書いてある通り 1/π
// 光源上にある場合、半球上一様サンプリングするので（下のほうでしている）pdf(ω) = 1 / 2πになり、
// これもpdf⊥が求めたいのでcosθで割る
// スペキュラ面上の場合、とりあえず1を返す（結局後で消滅する）
inline double pass_generation_probability(const Vertex &v0, const Vertex &v1) {
	switch (spheres[v0.id].ref_type) {
	case DIFFUSE: {
		if (v0.id != LightID) 
			return 1.0 / PI;
		else {
			const Vec dir = Normalize(v1.position - v0.position);
			return 1.0 / (2.0 * PI * Dot(v0.normal, dir));
		}
	} break;
	case SPECULAR: {
		return 1.0;
	} break;
	case REFRACTION: {
		return 1.0;
	} break;
	}
}

// ジオメトリ項を計算する
inline double geometry_term(const Vertex &v0, const Vertex &v1) {
	double t; // レイからシーンの交差位置までの距離
	int id;   // 交差したシーン内オブジェクトのID

	// v0 <-> v1 の可視判定
	const double dist = (v1.position - v0.position).LengthSquared();
	const Vec to0 = Normalize(v1.position - v0.position);
	intersect_scene(Ray(v0.position, to0), &t, &id);

	if (Dot(to0, v0.normal) >= 0 && fabs(sqrt(dist) - t) < EPS) {
		const double c0 = Dot(v0.normal, Normalize(v1.position - v0.position));
		const double c1 = Dot(v1.normal, Normalize(v0.position - v1.position));
		return c0 * c1 / dist;
	} else {
		return 0.0;
	}
}
 
// ジオメトリ項の計算は重いのでキャッシュする
struct GeometryTermCache {
	std::vector<Vertex*> indexs;
	std::vector<double> cache;
	
	GeometryTermCache(std::vector<Vertex>& light_vertices, std::vector<Vertex>& eye_vertices) {
		int NL = light_vertices.size();
		int NE = eye_vertices.size();
		// indexsに L0, L1, ... L(NL-1), E(NE - 1), E(NE - 2) ... E(0) という順番で格納しておく
		// calcで計算するときはインデックスだけ指定してジオメトリ項を計算できるようにしておく
		for (int i = 0; i < NL; i ++)
 			indexs.push_back(&light_vertices[i]);
		for (int i = 0; i < NE; i ++)
			indexs.push_back(&eye_vertices[NE - 1 - i]);
		std::vector<double>((NL + NE) * (NL + NE), -1.0).swap(cache);
	}
	
	inline double calc(int idx0, int idx1) {
		int tmp0 = idx0;
		int tmp1 = idx1;
		if (cache[indexs.size() * tmp0 + tmp1] < 0.0) { // キャッシュになかった
			return (cache[indexs.size() * idx0 + idx1] = geometry_term(*indexs[tmp0], *indexs[tmp1]));
		}
		// キャッシュにあった
		return cache[indexs.size() * idx0 + idx1];
	}
};



// ray方向からの放射輝度を求める
Color radiance(const Ray &camera, const Ray &ray, const int depth, int *used_sample) {
	// まず視線からパスを生成していく
	// E0, E1, E2 ... 
	// 今回は (視点(カメラ原点)) ---> E0 ---> E1 ... というふうに定式化した
	std::vector<Vertex> eye_vertices;
	trace_scene(ray, 0, &eye_vertices);

	// 光源からパスを生成する
	// 光源上の一点をサンプリングする
	const double r1 = 2 * PI * rand01();
	const double r2 = 1.0 - 2.0 * rand01() ;
	const Vec light_pos = spheres[LightID].position + ((spheres[LightID].radius + EPS) * Vec(sqrt(1.0 - r2*r2) * cos(r1), sqrt(1.0 - r2*r2) * sin(r1), r2));

	const Vec normal  = Normalize(light_pos - spheres[LightID].position);
	// 光源上の点から半球サンプリングする（一様サンプリング）
	Vec w, u, v;
	w = normal;
	if (fabs(w.x) > 0.1)
		u = Normalize(Cross(Vec(0.0, 1.0, 0.0), w));
	else
		u = Normalize(Cross(Vec(1.0, 0.0, 0.0), w));	
	v = Cross(w, u);
	const double u1 = 2 * PI * rand01();
	const double u2 = rand01(), r2s = sqrt(r2);
	const Vec light_dir = Normalize((u * cos(u1) * sin(acos(1.0 - u2)) + v * sin(u1) * sin(acos(1.0 - u2)) + w * (1.0 - u2)));

	const Ray light_ray(light_pos, light_dir);
	// 光源からパスを生成していく
	// L0, L1, L2 ...
	std::vector<Vertex> light_vertices;
	// この頂点がL0に相当する
	light_vertices.push_back(Vertex(light_ray.org, 1.0 / (2.0 * PI * Dot(normal, light_dir)), LightID, Color(1.0, 1.0, 1.0), normal));
	trace_scene(light_ray, 0, &light_vertices);

	// 以下、頂点間をつないで光源から視点への光輸送を計算する
	// PA_lightは光源上のある点をサンプリングするときの確率密度
	const double PA_light = 1.0 / (4.0 * PI * pow(spheres[LightID].radius, 2.0));
	// PA_eyeは視点上の～確率密度だが今回は1.0（決定的）
	const double PA_eye = 1.0;
	
	const int NL = light_vertices.size();
	const int NE = eye_vertices.size();
	
	// 光源からの光輸送を計算しやすい形で保存しておく
	// 詳しくはhttp://graphics.stanford.edu/courses/cs348b-03/papers/veach-chapter10.pdf (10.6)
	std::vector<Color> alpha_L(NL + 1);
	alpha_L[0] = Color(1.0, 1.0, 1.0);
	if (NL >= 1)
		// Le(0) / PA(L0)
		alpha_L[1] = (PI  * spheres[LightID].emission) / PA_light ;
	if (NL >= 2)
		// Le(1) / ...
		alpha_L[2] = (1.0 / PI) / light_vertices[0].probability_to_next_vertex * alpha_L[1];
	for (int i = 3; i < NL + 1; i ++) {
		alpha_L[i] = Multiply(light_vertices[i - 2].brdf / light_vertices[i - 2].probability_to_next_vertex, alpha_L[i - 1]);
	}

	// 視点への光輸送を計算しやすい形で保存しておく
	// 詳しくはhttp://graphics.stanford.edu/courses/cs348b-03/papers/veach-chapter10.pdf (10.7)
	std::vector<Color> alpha_E(NE + 1);
	alpha_E[0] = Color(1.0, 1.0, 1.0);
	if (NE >= 1)
		// We(0) / PA(E0)
		alpha_E[1] = (1.0 / PA_eye) * Color(1.0, 1.0, 1.0);
	for (int i = 2; i < NE + 1; i ++) {
		alpha_E[i] = Multiply(eye_vertices[i - 2].brdf / eye_vertices[i - 2].probability_to_next_vertex, alpha_E[i - 1]);
	}

	// LightサブパスとEyeサブパスをつなげるときの、そのつなげるための係数
	std::vector<Color> c((NL + 1) * (NE + 1));
	// 以下、s, tはそれぞれ生成するパスに含まれるLightサブパス頂点数とEyeサブパス頂点数
	// s = 0, t = 0 のときはどうにもならない
	c[0] = 0.0;

	// s = 0: Lightサブパスにおける頂点数が0
	for (int t = 1; t < NE+1; t ++) {
		if (eye_vertices[t - 1].id == LightID)
			c[0 * (NE+1) + t] = spheres[LightID].emission;
		else
			c[0 * (NE+1) + t] = Color(0.0, 0.0, 0.0);
	}
	
	// t = 0: Eyeサブパスにおける頂点数が0
	for (int s = 1; s < NL+1; s ++) {
		// 今回スクリーンは当たり判定ないので全部0
		c[s * (NE+1) + 0] = Color(0.0, 0.0, 0.0);
	}
	
	GeometryTermCache gcache(light_vertices, eye_vertices);

	// s >= 1 かつ t >= 1: 一般の場合
	for (int s = 1; s < NL+1; s ++) {
		for (int t = 1; t < NE+1; t ++) {
			Color brdf0, brdf1;
			if (s == 1) {
				brdf0 = (1.0 / PI) * Color(1.0, 1.0, 1.0); // Le(1)
			} else {
				brdf0 = BRDF(light_vertices[s - 1].id, light_vertices[s - 1].position, 
									light_vertices[s - 1].position - light_vertices[s - 2].position,
									eye_vertices[t - 1].position - light_vertices[s - 1].position) * spheres[light_vertices[s - 1].id].color;
			}

			if (t == 1) {
				brdf1 = BRDF(eye_vertices[t - 1].id, eye_vertices[t - 1].position, 
									eye_vertices[t - 1].position - light_vertices[s - 1].position,
									camera.org - eye_vertices[t - 1].position) * spheres[eye_vertices[t - 1].id].color;
			} else {
				brdf1 = BRDF(eye_vertices[t - 1].id, eye_vertices[t - 1].position, 
									eye_vertices[t - 1].position - light_vertices[s - 1].position,
									eye_vertices[t - 2].position - eye_vertices[t - 1].position) * spheres[eye_vertices[t - 1].id].color;
			}

			const double G = gcache.calc(s - 1, NL + NE - t);

			c[s * (NE+1) + t] = G * Multiply(brdf0, brdf1);
		}
	}

	// 重みを求める
	// 詳しくはhttp://graphics.stanford.edu/courses/cs348b-03/papers/veach-chapter10.pdf (10.9)
	std::vector<double> weight((NL + 1) * (NE + 1), 0.0);
	for (int s = 0; s < NL + 1; s ++) {
		for (int t = 1; t < NE + 1; t ++) {
			std::vector<double> p(s + t + 1);
			const int k = s + t - 1;

			if (k == 0) {
				weight[s * (NE+1) + t] = 1.0;
				continue;
			}

			// GeometryCacheを使うためにidxの形でアクセスできるようにしておく
			// あまり頭良くない
			std::vector<int> idx;
			for (int i = 0; i < s; i ++)
				idx.push_back(i);
			for (int i = 0; i < t; i ++)
				idx.push_back(NL + (NE - t) + i);

			// xs = L0, L1, ... L(s-1), E(t-1), E(t-2), ... E(0) の順番で並べる
			std::vector<Vertex*> xs;
			for (int i = 0; i < s; i ++)
				xs.push_back(&light_vertices[i]);
			for (int i = 0; i < t; i ++)
				xs.push_back(&eye_vertices[t-1-i]);

			// 最初にpsを適当に決める。とりあえず1とかにしておく。
			// 結局pの比率だけが問題になるのでこれでよい。
			p[s] = 1.0;

			// s = 0: Lightサブパスの長さが0の場合はちょっと特殊なので別処理する
			if (s == 0) {
				p[1] = p[0] * PA_light / (pass_generation_probability(*xs[1], *xs[0]) * geometry_term(*xs[1], *xs[0]));
				for (int i = s + 1; i < k; i ++) {
					p[i + 1] = p[i] * gcache.calc(idx[i - 1], idx[i]) * pass_generation_probability(*xs[i - 1], *xs[i])
 									/(gcache.calc(idx[i + 1], idx[i]) * pass_generation_probability(*xs[i + 1], *xs[i]));
				}
				if (k - 1 >= 0) 
					p[k+1] = p[k] * gcache.calc(idx[k - 1], idx[k]) * pass_generation_probability(*xs[k - 1], *xs[k]) / PA_eye;
			} else {
				for (int i = s; i < k; i ++) {
					p[i + 1] = p[i] * gcache.calc(idx[i - 1], idx[i]) * pass_generation_probability(*xs[i - 1], *xs[i])
 									/(gcache.calc(idx[i + 1], idx[i]) * pass_generation_probability(*xs[i + 1], *xs[i]));
				}
				if (k - 1 >= 0) 
					p[k+1] = p[k] * gcache.calc(idx[k - 1], idx[k]) * pass_generation_probability(*xs[k - 1], *xs[k]) / PA_eye;

				for (int i = s - 1; i > 0; i --) {
					p[i] = p[i + 1] * gcache.calc(idx[i + 1], idx[i]) * pass_generation_probability(*xs[i + 1], *xs[i])
									/(gcache.calc(idx[i - 1], idx[i]) * pass_generation_probability(*xs[i - 1], *xs[i]));
				}
				if (s > 0) 
					p[0] = p[1] * gcache.calc(idx[1], idx[0]) * pass_generation_probability(*xs[1], *xs[0]) / PA_light;
			}
			
			// スペキュラが絡んだ場合、Dirac関数でうんたらなのでそういうのが絡むpは0.0にする
			// 詳しくはhttp://graphics.stanford.edu/courses/cs348b-03/papers/veach-chapter10.pdf Sec. 10.3.5
			for (int j = 0; j < k + 1; j ++) {
				if (spheres[xs[j]->id].ref_type == SPECULAR || spheres[xs[j]->id].ref_type == REFRACTION) {
					p[j] = 0.0;
					p[j+1] = 0.0;
				}
			}

			// Power Heuristic
			double w = 0.0;
			for (int i = 0; i < s + t + 1; i ++) {
				w += p[i] * p[i];
			}
			if (w != 0.0)
				weight[s * (NE+1) + t] = 1.0 / w;
		}
	}
	
	// 実際にパスをつなげて輝度を求めるところ
	// 今まで求めてきた項を積算するだけ
	Color accum;
	for (int s = 0; s < NL+1; s ++) {
		for (int t = 1; t < NE+1; t ++) {
			if (weight[s * (NE+1) + t] > 0.0)	
				(*used_sample) ++;
			accum = accum + weight[s * (NE+1) + t] * Multiply(Multiply(alpha_L[s], c[s * (NE+1) + t]), alpha_E[t]);
		}
	}
	return accum;
}

// *** .hdrフォーマットで出力するための関数 ***
struct HDRPixel {
	unsigned char r, g, b, e;
	HDRPixel(const unsigned char r_ = 0, const unsigned char g_ = 0, const unsigned char b_ = 0, const unsigned char e_ = 0) :
	r(r_), g(g_), b(b_), e(e_) {};
	unsigned char get(int idx) {
		switch (idx) {
		case 0: return r;
		case 1: return g;
		case 2: return b;
		case 3: return e;
		} return 0;
	}

};

// doubleのRGB要素を.hdrフォーマット用に変換
HDRPixel get_hdr_pixel(const Color &color) {
	double d = std::max(color.x, std::max(color.y, color.z));
	if (d <= 1e-32)
		return HDRPixel();
	int e;
	double m = frexp(d, &e); // d = m * 2^e
	d = m * 256.0 / d;
	return HDRPixel(color.x * d, color.y * d, color.z * d, e + 128);
}

// 書き出し用関数
void save_hdr_file(const std::string &filename, const Color* image, const int width, const int height) {
	FILE *fp = fopen(filename.c_str(), "wb");
	if (fp == NULL) {
		std::cerr << "Error: " << filename << std::endl;
		return;
	}
	// .hdrフォーマットに従ってデータを書きだす
	// ヘッダ
	unsigned char ret = 0x0a;
	fprintf(fp, "#?RADIANCE%c", (unsigned char)ret);
	fprintf(fp, "# Made with 100%% pure HDR Shop%c", ret);
	fprintf(fp, "FORMAT=32-bit_rle_rgbe%c", ret);
	fprintf(fp, "EXPOSURE=1.0000000000000%c%c", ret, ret);

	// 輝度値書き出し
	fprintf(fp, "-Y %d +X %d%c", height, width, ret);
	for (int i = height - 1; i >= 0; i --) {
		std::vector<HDRPixel> line;
		for (int j = 0; j < width; j ++) {
			HDRPixel p = get_hdr_pixel(image[j + i * width]);
			line.push_back(p);
		}
		fprintf(fp, "%c%c", 0x02, 0x02);
		fprintf(fp, "%c%c", (width >> 8) & 0xFF, width & 0xFF);
		for (int i = 0; i < 4; i ++) {
			for (int cursor = 0; cursor < width;) {
				const int cursor_move = std::min(127, width - cursor);
				fprintf(fp, "%c", cursor_move);
				for (int j = cursor;  j < cursor + cursor_move; j ++)
					fprintf(fp, "%c", line[j].get(i));
				cursor += cursor_move;
			}
		}
	}

	fclose(fp);
}

int main(int argc, char **argv) {
	int width = 640;
	int height = 480;
	int samples = 16;

	// カメラ位置
	Ray camera(Vec(50.0, 52.0, 295.6), Normalize(Vec(0.0, -0.042612, -1.0)));
	// シーン内でのスクリーンのx,y方向のベクトル
	Vec cx = Vec(width * 0.5135 / height);
	Vec cy = Normalize(Cross(cx, camera.dir)) * 0.5135;
	Color *image = new Color[width * height];
	
 #pragma omp parallel for schedule(dynamic, 1)
	for (int y = 0; y < height; y ++) {
		int used_sample = 0;
		for (int x = 0; x < width; x ++) {
			int image_index = y * width + x;	
			image[image_index] = Color();

			// 2x2のサブピクセルサンプリング
			for (int sy = 0; sy < 2; sy ++) {
				for (int sx = 0; sx < 2; sx ++) {
					Color accumulated_radiance = Color();

					// 一つのサブピクセルあたりsamples回サンプリングする
					for (int s = 0; s < samples; s ++) {
						// テントフィルターによってサンプリング
						// ピクセル範囲で一様にサンプリングするのではなく、ピクセル中央付近にサンプルがたくさん集まるように偏りを生じさせる
						const double r1 = 2.0 * rand01(), dx = r1 < 1.0 ? sqrt(r1) - 1.0 : 1.0 - sqrt(2.0 - r1);
						const double r2 = 2.0 * rand01(), dy = r2 < 1.0 ? sqrt(r2) - 1.0 : 1.0 - sqrt(2.0 - r2);
						Vec dir = cx * (((sx + 0.5 + dx) / 2.0 + x) / width - 0.5) +
								  cy * (((sy + 0.5 + dy) / 2.0 + y) / height- 0.5) + camera.dir;
						accumulated_radiance = accumulated_radiance + 
							radiance(camera, Ray(camera.org + dir * 130.0, Normalize(dir)), 0, &used_sample) / samples;
					}
					
					image[image_index] = image[image_index] + accumulated_radiance;
				}
			}
		}
		std::cerr << "Rendering (average: " << ((double)used_sample / width) << " spp) " << (100.0 * y / (height - 1)) << "%" << std::endl;
	}
	
	// .hdrフォーマットで出力
	save_hdr_file(std::string("image.hdr"), image, width, height);
}