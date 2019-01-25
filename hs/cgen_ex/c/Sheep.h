#ifndef CGEN_SHEEP_H
#define CGEN_SHEEP_H

#include <Sheep.h>
#include <Dog.h>
#include <Animal.h>

extern "C"
{

using namespace Animals;



#ifdef CGEN_HS
#endif

void Animals_Sheep_delete(Sheep* this_ptr);
Sheep* Animals_Sheep_new(int wooliness_level_);
void Animals_Sheep_make_sound(Sheep* this_ptr);
void Animals_Sheep_shear(Sheep* this_ptr);

}

#endif

