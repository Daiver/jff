#ifndef CGEN_DOG_H
#define CGEN_DOG_H

#include <Sheep.h>
#include <Dog.h>
#include <Animal.h>

extern "C"
{

using namespace Animals;



#ifdef CGEN_HS
#endif

void Animals_Dog_delete(Dog* this_ptr);
Dog* Animals_Dog_new();
void Animals_Dog_make_sound(Dog* this_ptr);

}

#endif

