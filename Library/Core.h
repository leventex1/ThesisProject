#pragma once

#define namespace_start namespace mogi {
#define namespace_end }

#ifdef EXPORT_LIBRARY
#define LIBRARY_API __declspec(dllexport)
#else
#define LIBRARY_API __declspec(dllimport)
#endif // EXPORT_LIBRARY
