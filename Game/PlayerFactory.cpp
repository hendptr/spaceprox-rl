#include "Player.h"

namespace SimpleGame7
{
__declspec(noinline) Player* CreatePlayer()
{
    static Player player;
    return &player;
}
} // namespace SimpleGame7