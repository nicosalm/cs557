#pragma once

#define XDIM 16384
#define YDIM 16384

// at this point treat as 2d arrays and that will be supported just fine
void ComputeLaplacian(const float (&u)[XDIM][YDIM], float (&Lu)[XDIM][YDIM]);

// loop order swapped (Y before X), everything gets fucked
// 22ms -> 50ms
// reasons: spatial locality reduction!
//          cache lines
