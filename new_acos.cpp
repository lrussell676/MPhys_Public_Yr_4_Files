#include <iostream> 
#include <cmath>
#include <vector>
#include <chrono>

double acos_calc_raw(int value_calc_size, double* acos_calc_values){

  double ans, raw_acos_calc_time;

  auto begin = std::chrono::system_clock::now();

  for (int i=0; i<value_calc_size; i++) {
    ans = acos(acos_calc_values[i]);
   // if (i == 0 || i == 1 || i == 2 || i == value_calc_size-3 || i == value_calc_size-2 || i == value_calc_size-1) {
   // printf("\n%f\n", ans);
   // }
  }

  auto end = std::chrono::system_clock::now();
  std::chrono::duration<double, std::milli> elapsed = end - begin;
  raw_acos_calc_time = elapsed.count();
  std::cout << "acos_calc_raw time = " << raw_acos_calc_time << "ms" << std::endl;

  return raw_acos_calc_time;
}

/* ----------- acos_look_up (array, y values) ----------- */

/*
int acos_calc_lookup(int value_calc_size, double* acos_calc_values){
for (int i=0; i<value_calc_size; i++) {
acos_calc_values[i] = acos(acos_calc_values[i]);
//if (i == 0 || i == 1 || i == 2 || i == value_calc_size-3 || i == value_calc_size-2 || i == value_calc_size-1) {
//  printf("\n%f\n", acos_calc_values[i]);
}
return 0;
} */

/* ----------- acos_look_up (vector, y values) ----------- */

int acos_calc_lookup(int value_calc_size, std::vector<std::vector<double>>& acos_calc_values){

  auto begin = std::chrono::system_clock::now();

  for (int i=0; i<value_calc_size; i++) {
    acos_calc_values[1][i] = acos(acos_calc_values[0][i]);
    //if (i == 0 || i == 1 || i == 2 || i == value_calc_size-3 || i == value_calc_size-2 || i == value_calc_size-1) {
    //  printf("\n%f\n", acos_calc_values[1][i]);
    //}
  }

  auto end = std::chrono::system_clock::now();
  std::chrono::duration<double, std::milli> elapsed = end - begin;
  std::cout << "acos loop-up y-generation time = " << elapsed.count() << "ms" << std::endl;

  return 0;
}

/* ------------------------------------------------------ */

int acos_cubic_spline_all_values(
  int look_up_size, int value_calc_size, 
    double* acos_calc_values, std::vector<std::vector<double>>& acos_look_up,
    double& spline_acos_calc_time){

  double x, y, ya, yb, xa, xb;
  double index_float;
  int index_a, index_b;
  auto begin = std::chrono::system_clock::now();
  begin = std::chrono::system_clock::now();

  for (int i=0; i<=value_calc_size; i++) {
    x = acos_calc_values[i];
	//x = 0;
    index_float = (x+1)*((look_up_size-1)*0.5)+1;
	index_a = floor(index_float);
    index_b = ceil(index_float);
	if ( (index_a - index_b) == 0 ) {
	  y = acos_look_up[1][index_a];
	} else {
	  xa = acos_look_up[0][index_a];
      xb = acos_look_up[0][index_b];
      ya = acos_look_up[1][index_a];
      yb = acos_look_up[1][index_b];
      y = ya + ((yb - ya)/(xb - xa)) * (x - xa);	
	}	  

    //if (i == 0 || i == (value_calc_size-1)/4 || i == (value_calc_size-1)/2 || i == 3*(value_calc_size-1)/4 || i == value_calc_size-1) {
    //  printf("spline at %f is %f \n", x, y);
    //} 
  }

  auto end = std::chrono::system_clock::now();
  end = std::chrono::system_clock::now();
  std::chrono::duration<double, std::milli> elapsed = end - begin;
  elapsed = end - begin;
  spline_acos_calc_time = elapsed.count();
  std::cout << "spline calc time = " << spline_acos_calc_time << "ms" << std::endl;

  return 0;	
}

/* ------------------------------------------------------------------ */
/* ------------------------------------------------------------------ */
/* ------------------------------------------------------------------ */

int main(){

  printf("\n/--------------------------------------/\n\n");

  auto begin = std::chrono::system_clock::now();
  auto end = std::chrono::system_clock::now();

  // initialisation 

  int look_up_size = 10001;
  int value_calc_size = 1000000;

  std::vector<std::vector<double>> acos_look_up(2);
  acos_look_up[0].resize(look_up_size, 0.0);
  acos_look_up[1].resize(look_up_size, 0.0);
  double look_up_increment = 1.0/(look_up_size-1); 

  /*   double acos_look_up[2][look_up_size];
  double look_up_increment = 1.0/(look_up_size-1); */

  double acos_calc_values[value_calc_size];
  double calc_values_increment = 1.0/(value_calc_size-1);

  double spline_acos_calc_time;

  /* ----------- acos_look_up_x (vector, x linspace) ----------- */

  begin = std::chrono::system_clock::now();

  acos_look_up[0][0] = -1.0;
  for (int i=1; i<look_up_size-1; i++) {
    acos_look_up[0][i] = acos_look_up[0][i-1] + 2*look_up_increment;
  }
  acos_look_up[0][look_up_size-1] = 1.0;

  end = std::chrono::system_clock::now();
  std::chrono::duration<double, std::milli> elapsed = end - begin;
  std::cout << "acos loop-up x-generation time = " << elapsed.count() << "ms" << std::endl;

  /* ----------- acos_calc_values_x (x linspace) ----------- */

  begin = std::chrono::system_clock::now();

  acos_calc_values[0] = -1.0;
  for (int i=1; i<value_calc_size-1; i++) {
    acos_calc_values[i] = acos_calc_values[i-1] + 2*calc_values_increment;
  }
  acos_calc_values[value_calc_size-1] = 1.0;

  end = std::chrono::system_clock::now();
  elapsed = end - begin;
  std::cout << "raw calc values x-generation time = " << elapsed.count() << "ms" << std::endl;

  /* ----------- functions for calculations ----------- */

  double raw_acos_calc_time = 
  acos_calc_raw(value_calc_size, acos_calc_values);  // raw calc of acos values

  acos_calc_lookup(look_up_size, acos_look_up);        // look-up table calc (y values)                  

  acos_cubic_spline_all_values(                        // spline creation and acos calc
  look_up_size, value_calc_size, 
  acos_calc_values, acos_look_up, spline_acos_calc_time);

  printf("\n/--------------------------------------/\n\n");

  double ratio = (spline_acos_calc_time / raw_acos_calc_time);

  printf("Therefore, spline takes around %fx the time of raw calculation of acos\n\n", ratio);

  /* ----------- extra print statements ----------- */

  /* printf("\n%f  %f  %f  %f  %f  %f \n", 
  acos_look_up[0], acos_look_up[1], acos_look_up[2], 
  acos_look_up[look_up_size-3], acos_look_up[look_up_size-2], acos_look_up[look_up_size-1]);
  printf("%f  %f  %f  %f  %f  %f \n \n", 
  acos_calc_values[0], acos_calc_values[1], acos_calc_values[2], 
  acos_calc_values[value_calc_size-3], acos_calc_values[value_calc_size-2], acos_calc_values[value_calc_size-1]); */

  /* printf("%f  %f  %f  %f  %f  %f \n\n", 
  acos_look_up[0][0], acos_look_up[0][1], acos_look_up[0][2], 
  acos_look_up[0][look_up_size-3], acos_look_up[0][look_up_size-2], acos_look_up[0][look_up_size-1]);

  /* printf("%f  %f  %f  %f  %f  %f \n\n", 
  acos_look_up[1][0], acos_look_up[1][1], acos_look_up[1][2], 
  acos_look_up[1][look_up_size-3], acos_look_up[1][look_up_size-2], acos_look_up[1][look_up_size-1]);  */

  return 0;
}