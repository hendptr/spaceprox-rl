#include "Player.h"

namespace SpaceProx
{
__declspec(noinline) Player* CreatePlayer()
{
    static Player player;
    return &player;
}
} // namespace SpaceProx